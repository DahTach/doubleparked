import numpy as np
import itertools
from numpy.polynomial import polynomial as P
import math

import networkx
from networkx.algorithms.components.connected import connected_components

from sklearn.cluster import DBSCAN

from drawing import Draw

draw = Draw()


class Car:
    def __init__(self, id, confidence, bbox) -> None:
        self.id = id
        self.confidence = confidence
        # bbox = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        self.bbox = bbox
        self.bl = bbox[0]  # bottom left corner
        self.br = bbox[1]  # bottom right corner
        self.tr = bbox[2]  # top right corner
        self.tl = bbox[3]  # top left corner
        self.corners = self.bl, self.br, self.tr, self.tl
        self.bl_lines = []
        self.br_lines = []
        self.tr_lines = []
        self.tl_lines = []
        self.lines = self.bl_lines, self.br_lines, self.tr_lines, self.tl_lines
        self.bl_angles = []
        self.br_angles = []
        self.tr_angles = []
        self.tl_angles = []
        self.angles = self.bl_angles, self.br_angles, self.tr_angles, self.tl_angles
        self.score = None


class Cars:
    def __init__(self, predictions) -> None:
        self.cars = []
        self.groups = []
        self.groups_angles = {
            # group_index: [angles]
        }
        self.db_parked = []
        self.get_cars(predictions)
        self.get_connections()
        self.get_angles()

    def add_car(self, car):
        if not isinstance(car, Car):
            raise TypeError("Can only add Car objects")
        self.cars.append(car)

    def __getitem__(self, index):
        """Allows indexing to access cars"""
        return self.cars[index]

    def get_cars(self, predictions):
        boxes = predictions[0].boxes.xyxy.tolist()
        classes = predictions[0].boxes.cls.tolist()
        confidences = predictions[0].boxes.conf.tolist()

        for i, box in enumerate(boxes):
            if classes[i] == 1:
                x0 = int(box[0])
                x1 = int(box[2])
                y0 = int(box[1])
                y1 = int(box[3])

                # xc = x1 - ((x1 - x0) / 2)
                # yc = y1 - ((y1 - y0) / 2)

                bbox = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                self.add_car(Car(i, confidences[i], bbox))

    def get_connections(self):
        """
        Args:
            cars = {
                car_index: [(x0, y0),(x1, y0),(x1, y1), (x0, y1)]
            }

        Returns:
            cars_lines = {
                car_index: [ [] for corner_index in range(4) ]
            }
        """
        cars_groups = []

        for car1, car2 in itertools.combinations(self.cars, 2):
            if self.are_intersected(car1.bbox, car2.bbox) or self.are_close(
                car1.bbox, car2.bbox
            ):
                cars_groups.append([car1.id, car2.id])
                threshold = self.calc_connection_range(car1.bbox, car2.bbox)

                for corner in range(4):
                    if (
                        max(car1.corners[corner][0], car2.corners[corner][0])
                        - min(car1.corners[corner][0], car2.corners[corner][0])
                        <= threshold
                    ):
                        car1.lines[corner].append(
                            (car1.corners[corner], car2.corners[corner])
                        )
                        car2.lines[corner].append(
                            (car2.corners[corner], car1.corners[corner])
                        )

        self.groups = self.merge_sublist_with_commons(cars_groups)

    def get_angles(self):
        for car in self.cars:
            for i, corner in enumerate(car.lines):
                for line in corner:
                    angle = self.calc_line_angle(line)
                    car.angles[i].append(angle)
                    for gi, group in enumerate(self.groups):
                        if car.id in group:
                            self.groups_angles.setdefault(gi, [angle])
                            self.groups_angles[gi].append(angle)

    def calc_line_angle(self, line, rounding=1) -> float:
        """
        Args:
            line: tuple of points (x,y),(x,y)

        Returns:
            angle: angle normalized to 360deg and rounded
        """
        p1, p2 = line
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle = math.degrees(math.atan2(dy, dx))
        angle = self.normalize_angle(angle)

        return round(angle, rounding)

    def normalize_angle(self, angle) -> float:
        """Normalizes an angle to be between 0 and 360 degrees.

        Args:
            angle: The angle in degrees.

        Returns:
            The normalized angle between 0 and 360 degrees.
        """

        # Reduce the angle to the range [-180, 180)
        angle = math.fmod(angle, 360)
        if angle < -180:
            angle += 360

        # Convert to the range [0, 360)
        return angle % 360

    def merge_sublist_with_commons(self, parent_list):
        """
        Args:
            parent_list = [['a','b','c'],['b','d','e'],['k'],['o','p'],['e','f'],['p','a'],['d','g']]
        Returns:
            merged_parent = [['a', 'c', 'b', 'e', 'd', 'g', 'f', 'o', 'p'], ['k']]
        """

        def to_graph(parent_list):
            G = networkx.Graph()
            for part in parent_list:
                # each sublist is a bunch of nodes
                G.add_nodes_from(part)
                # it also imlies a number of edges:
                G.add_edges_from(to_edges(part))
            return G

        def to_edges(parent_list):
            """
            treat `parent_list` as a Graph and returns it's edges
            to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
            """
            it = iter(parent_list)
            last = next(it)

            for current in it:
                yield last, current
                last = current

        G = to_graph(parent_list)
        merged = connected_components(G)

        subs = []

        for sub in merged:
            subs.append(list(sub))

        return subs

    def calc_connection_range(self, car1, car2) -> float:
        h1 = car1[2][1] - car1[0][1]
        w1 = car1[1][0] - car1[0][0]

        h2 = car2[2][1] - car2[0][1]
        w2 = car2[1][0] - car2[0][0]
        max_range = max(w1, h1, w2, h2)
        return max_range

    def are_intersected(self, rect1, rect2):
        left1, top1, right1, bottom1 = (
            rect1[0][0],
            rect1[0][1],
            rect1[2][0],
            rect1[2][1],
        )
        left2, top2, right2, bottom2 = (
            rect2[0][0],
            rect2[0][1],
            rect2[2][0],
            rect2[2][1],
        )

        return not (
            right1 <= left2  # Rect1 is left of Rect2
            or left1 >= right2  # Rect1 is right of Rect2
            or bottom1 <= top2  # Rect1 is above Rect2
            or top1 >= bottom2
        )

    def are_close(self, rect1, rect2):
        left1, top1, right1, bottom1 = (
            rect1[0][0],
            rect1[0][1],
            rect1[2][0],
            rect1[2][1],
        )
        left2, top2, right2, bottom2 = (
            rect2[0][0],
            rect2[0][1],
            rect2[2][0],
            rect2[2][1],
        )

        distance_x = min(
            abs(left2 - left1),
            abs(left2 - right1),
            abs(right2 - left1),
            abs(right2 - right1),
        )

        distance_y = min(
            abs(bottom2 - top1),
            abs(bottom2 - bottom1),
            abs(top2 - bottom1),
            abs(top2 - top1),
        )

        relative_size_y = min((top2 - bottom2), (top1 - bottom1))

        relative_size_x = min((right2 - left2), (right1 - left1)) / 2

        return distance_x <= relative_size_x and distance_y <= relative_size_y

    def double_parked(self):
        # Look for the car that has the most outliers in the lines angle grouping
        # I need all angles and the 4 corner outliers
        max_score = 0
        max_id = None
        second_score = 0
        second_id = None
        max_group = 0
        for car in self.cars:
            score = 0
            for i, group in enumerate(self.groups):
                max_group = max(max_group, len(group))
                if car.id in group:
                    for corner in car.angles:
                        for angle in corner:
                            score += self.angle_score(self.groups_angles[i], angle)
            car.score = int(score)
            print(f"car {car.id} - {car.score}")
            if score > max_score:
                second_score = max_score
                second_id = max_id
                max_score = score
                max_id = car.id

        # FIX: some scores are fucked up

        if max_group >= 4:
            self.db_parked.append(max_id)
            if (max_score - second_score) < (max_score / 10):
                self.db_parked.append(second_id)

    def angle_score(self, angles, angle, tolerance=15, min=1) -> int:
        """Checks how good the angle fits in the group angles
        Args:
            angles: list of angles normalized to 360deg and rounded to 1 decimal
            angle: the angle (normalized to 360deg and rounded to 1 decimal) that I want to evaluate

        How it works:
            Group angles into clusters.
            Angles in smaller clusters or farther away from cluster centers are more likely to be outliers.

        Returns:
            score: a score representative of how good the angle fits in the angles group
        """

        angles_reshaped = [[a] for a in angles]  # Reshape for DBSCAN
        clustering = DBSCAN(eps=tolerance, min_samples=min).fit(
            angles_reshaped
        )  # Adjust parameters

        labels = clustering.labels_
        if angle in angles:
            angle_label = labels[angles.index(angle)]
            cluster_size = (labels == angle_label).sum()

            # Map cluster size to score (smaller cluster size -> lower score)
            score = 100 / cluster_size
        else:
            score = 0  # Angle not in existing groups -> likely outlier

        return score

    def draw_cars(self, img) -> None:
        for car in self.cars:
            for corner in car.lines:
                for line in corner:
                    draw.line(img, line)
            start_point = (int(car.bbox[0][0]), int(car.bbox[0][1]))
            end_point = (int(car.bbox[2][0]), int(car.bbox[2][1]))
            draw.rectangle(img, start_point, end_point, text=f"car {car.id}")
            if car.id in self.db_parked:
                draw.rectangle(
                    img, start_point, end_point, text=f"car {car.id}", transparent=True
                )
