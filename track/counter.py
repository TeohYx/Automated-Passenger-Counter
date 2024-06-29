import datetime
import cv2
import time
import math

class Counter:
    def __init__(self):
        self.passenger_in = 0
        self.passenger_out = 0
        self.iden_standby = []  # (iden, in/out) eg: (30, 1) in-0; out-1; middle-2 (means appear in the line randomly)
        self.intersect_in = None
        self.intersect_out = None

    def get_passenger_in(self):
        return self.passenger_in

    def get_passenger_out(self):
        return self.passenger_out

    def get_iden_standby(self):
        return self.iden_standby

    def check_intersect(self, segments, image, track_info_obj, mongodb, doc_time):
        """
        Algorithm to check if passenger boarding or alighting the public bus.

        Return True if any passenger boarded or alighted the bus, and False if none does
        """
        def distance_between_points(x1, y1, x2, y2):
            return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        def is_point_in_triangle(A, B, C, P):
            def area(X, Y, Z):
                return abs((X[0] * (Y[1] - Z[1]) + Y[0] * (Z[1] - X[1]) + Z[0] * (X[1] - Y[1])) / 2.0)

            # Calculate area of the triangle ABC
            A_ABC = area(A, B, C)
            
            # Calculate area of the triangles PAB, PBC, PCA
            A_PAB = area(P, A, B)
            A_PBC = area(P, B, C)
            A_PCA = area(P, C, A)

            # Check if sum of PAB, PBC and PCA is same as ABC
            return A_ABC == A_PAB + A_PBC + A_PCA

        def is_point_inside_area(A, B, C, D, P):
            points = [A, B, C, D]
            points.sort(key=lambda point: (point[1], point[0])) # Sort points based on y-coordinate first, then x-coordinate

            top_left = points[0]
            points = points[1:]

            points.sort(key=lambda point: (-point[1], -point[0]))
            bottom_right = points[0]

            points = points[1:]

            points.sort(key=lambda point: (point[0], point[1]))
            bottom_left = points[0]
            top_right = points[1]

            # Check if the point P is inside either of the two triangles ABD and BCD
            return is_point_in_triangle(top_left, top_right, bottom_left, P) or is_point_in_triangle(top_right, bottom_right, bottom_left, P)

        def calculate_intersection(x1, y1, x2, y2, xp, yp, xc, yc):
            """
            Calculate the intersection point of the line segment formed by (xp, yp) and (xc, yc) with the line segment (x1, y1, x2, y2)

            Return: True if intersected, and False if not intersected
            """
            dx1 = xc - xp
            dy1 = yc - yp
            dx2 = x2 - x1
            dy2 = y2 - y1
            denom = dx1 * dy2 - dy1 * dx2

            if denom == 0:
                # No intersection, the lines are parallel or coincident
                # print("No intersect")
                return False, 0
            else:
                # Calculate intersection point
                ua = (dx2 * (yp - y1) - dy2 * (xp - x1)) / denom
                ub = (dx1 * (yp - y1) - dy1 * (xp - x1)) / denom
                if 0 <= ua <= 1 and 0 <= ub <= 1:
                    xi = xp + ua * dx1
                    yi = yp + ua * dy1
                    # print("Intersected")
                    # return xi, yi
                    # print((xi, yi))
                    return True, (xi, yi)
                else:
                    # Intersection point is outside the line segment (x1, y1, x2, y2)
                    # print("Intersection point outside")
                    return False, 0

        is_record = False
        if segments is not None:
            for seg in segments:
                print(f"Standby got {self.get_iden_standby()}")
                # print(seg)
                # result = (dirention, iden)

                # point is intersected point
                inner, inner_point = calculate_intersection(track_info_obj.inner_line[0][0], track_info_obj.inner_line[0][1], track_info_obj.inner_line[0][2], track_info_obj.inner_line[0][3], seg[1], seg[2], seg[3], seg[4])
                if inner_point != 0:
                    self.intersect_in = inner_point
                outer, outer_point = calculate_intersection(track_info_obj.outer_line[0][0], track_info_obj.outer_line[0][1], track_info_obj.outer_line[0][2], track_info_obj.outer_line[0][3] , seg[1], seg[2], seg[3], seg[4])
                if outer_point != 0:
                    self.intersect_out = outer_point

                # Scenario when the point is appear in between the lines - Track it, either board or alight, happen due to bad detection
                is_in_lines = is_point_inside_area(
                    (track_info_obj.inner_line[0][0], track_info_obj.inner_line[0][1]), 
                    (track_info_obj.inner_line[0][2], track_info_obj.inner_line[0][3]), 
                    (track_info_obj.outer_line[0][0], track_info_obj.outer_line[0][1]), 
                    (track_info_obj.outer_line[0][2], track_info_obj.outer_line[0][3]), 
                    (seg[1], seg[2])
                )
                standby_tag = [iden[1] for iden in self.get_iden_standby() if iden[0] == seg[0]]    # Get the tag in the standby area
                if standby_tag:
                    standby_tag = standby_tag[0]
                else:
                    standby_tag = -1
                if is_in_lines and standby_tag != 0 and standby_tag != 1:
                    if not inner and not outer:
                        # Appeared in the lines, register it in self.iden_standby
                        if (seg[0], 2) not in self.iden_standby:
                            self.iden_standby.append((seg[0], 2))
                            continue
                    elif inner:
                        # Boarding
                        cv2.putText(image, f"{seg[0]} boarded", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        is_record = True
                        if (seg[0], 2) in self.iden_standby: 
                            self.iden_standby.remove((seg[0], 2)) # Remove it if it is registered
                        self.passenger_in += 1
                        mongodb.increment_in(doc_time, 1) 
                        continue
                    elif outer:                     
                        cv2.putText(image, f"{seg[0]} alighted", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        print("here")
                        is_record = True
                        if (seg[0], 2) in self.iden_standby:
                            self.iden_standby.remove((seg[0], 2))
                        self.passenger_out += 1
                        mongodb.increment_out(doc_time, 1)
                        continue
                    else:
                        pass

                #Scenario when object pass thru both line in one frame
                if inner and outer:
                    # Check distance between intersect point and previous frame center
                    inner_dist = distance_between_points(seg[1], seg[2], self.intersect_in[0], self.intersect_in[1])
                    outer_dist = distance_between_points(seg[1], seg[2], self.intersect_out[0], self.intersect_out[1])                            
                    
                    # If closer in inner, means alighting, else boarding
                    if inner_dist < outer_dist:
                        # Alighting
                        cv2.putText(image, f"{seg[0]} alighted", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        is_record = True
                        self.passenger_out += 1
                        mongodb.increment_out(doc_time, 1)
                    elif inner_dist > outer_dist:
                        # Boarding
                        cv2.putText(image, f"{seg[0]} boarded", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        is_record = True
                        self.passenger_in += 1
                        mongodb.increment_in(doc_time, 1)                          
                    else:
                        print("Two line are overlapped")
                    continue
                        # two line are overlapped

                # Scenario when the object is not in standby area - Either not related, or possible boarding and alighting
                if not any(iden[0] == seg[0] for iden in self.get_iden_standby()):   # Check if iden is in standby area
                    # iden is not in standby area and is tracked
                    if inner:
                        self.iden_standby.append((seg[0], 1)) # 1 represent out (person touch the inner line and possibly going out)
                        # print(f"{seg[0]}:inner standby to outer")
                    elif outer:
                        self.iden_standby.append((seg[0], 0)) # 0 represent in (person touch the outer line and possibly going in)
                        # print(f"{seg[0]}:outer standby to inner")
                    else:
                        pass
                        # print(f"{seg[0]}:not related")
                    continue

                # Scenario when the object is in standby area - Either boarding, alighting or remain in standby zone
                for iden in self.get_iden_standby(): 
                    if iden[0] == seg[0]:
                        if inner: # Boarding
                            if iden[1] == 1:
                                continue
                            cv2.putText(image, f"{seg[0]} boarded", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            is_record = True
                            try:
                                self.iden_standby.remove((seg[0], 0))
                            except:
                                print("Identifier unable to remove")
                            self.passenger_in += 1
                            mongodb.increment_in(doc_time, 1)
                            break
                            # print(f"{seg[0]}: boarded")
                        elif outer: # Alighting
                            if iden[1] == 0:
                                continue
                            cv2.putText(image, f"{seg[0]} alighted", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            is_record = True
                            try:
                                self.iden_standby.remove((seg[0], 1))
                            except:
                                print("Identifier unable to remove")
                            self.passenger_out += 1
                            mongodb.increment_out(doc_time, 1)
                            break
                            # print(f"{seg[0]}: alighted")
                        else:
                            pass
                            # print(f"{seg[0]}: standby")
        return is_record

    @staticmethod
    def update_frame(info_last_frame, info_current_frame):
        """
        info_last_frame, info_current_frame both in form of [center (x,y), identifier]

        Return: prevoius and current point in list [identifier, x1, y1, x2, y2]
        The connection of these two centers check if the passenger passed the inner or outer line in later function.
        """
        # print(f"Last frame: {info_last_frame}")
        # print(f"Current frame: {info_current_frame}")
        
        segments = []

        if info_last_frame is None:
            print("No tracking was detected")
            return

        '''
        Compare last frame and current frame. If the identifier between two frame are same, their center are store in segment variable as a list.
        '''
        for _, last_frame in enumerate(info_last_frame):
            for _, curr_frame in enumerate(info_current_frame):
                if last_frame[1] == curr_frame[1]:
                    # print(last_frame)
                    segment = last_frame[1], last_frame[0][0], last_frame[0][1], curr_frame[0][0], curr_frame[0][1]
                    segments.append(segment)

        return segments

