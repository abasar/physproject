#!/usr/bin/env python3
import click
import cv2
import numpy as np
import scipy
import math
from dataclasses import dataclass

Point = tuple[int, int]


# arbitrary
GAUSSIAN_BLUR_KERNEL_SIZE = 5
# 200 is impossible because it is not between -pi and pi
IMPOSSIBLE_ANGLE = 200
# found from trial and error
ANGLE_INCREMENT = 0.1
# somewhat arbitrary
SAFETY_PIXELS = 50


class CouldntFindCorners(Exception):
    pass


class InvalidNumberOfPeaks(Exception):
    pass


@dataclass
class Gradients:
    magnitudes: np.typing.NDArray[np.float64]
    angles: np.typing.NDArray[np.float64]


@dataclass
class Template:
    angle: float
    distance1: float
    distance2: float


@dataclass
class Rectangle:
    min_x: int
    max_x: int
    min_y: int
    max_y: int


def load_image(filepath: str) -> np.ndarray:
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found: {filepath}")
    return image


def reduce_image_background(
    image: np.typing.NDArray[np.uint8],
) -> np.typing.NDArray[np.float64]:
    # we assume that most of the image is background, therefore the median
    # should give us approximately the intensity of the background
    median = np.median(image)
    no_bg = image.astype(np.float64) - median
    no_bg[no_bg < 0] = 0
    return no_bg


def apply_gaussian_blur(
    image: np.typing.NDArray[np.float64], kernel_size: int
) -> np.typing.NDArray[np.float64]:
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def find_gradients(image: np.typing.NDArray[np.float64]) -> Gradients:
    horizontal_sobel_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    vertical_sobel_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    horizontal_derivative = scipy.ndimage.convolve(image, horizontal_sobel_kernel)
    vertical_derivative = scipy.ndimage.convolve(image, vertical_sobel_kernel)

    gradient_magnitudes = np.sqrt(
        horizontal_derivative * horizontal_derivative
        + vertical_derivative * vertical_derivative
    )
    gradient_angles = np.atan2(vertical_derivative, horizontal_derivative)

    return Gradients(gradient_magnitudes, gradient_angles)


def rotate_image(
    image: np.typing.NDArray[np.uint8], angle: float
) -> np.typing.NDArray[np.uint8]:
    """Rotates an image by a given angle around its center."""
    h, w = image.shape
    center = (w // 2, h // 2)  # Compute the center

    # Compute the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, np.degrees(angle), 1.0)

    # Determine the new bounding dimensions
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust the rotation matrix to take translation into account
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    rotated_image = cv2.warpAffine(
        image,
        rotation_matrix,
        (new_w, new_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return rotated_image


def find_largest_contour(
    image: np.typing.NDArray[np.uint8],
) -> np.typing.NDArray[np.int32]:
    contours, _ = cv2.findContours(
        image.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    return max(contours, key=cv2.contourArea)


def prompt_user_for_template_corners(image: np.typing.NDArray[np.uint8]) -> list[Point]:
    clean_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = np.copy(clean_image)
    points = []
    window_title = "Please select corners of the template"

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(window_title, image)
            points.append((x, y))

        if event == cv2.EVENT_RBUTTONDOWN:
            image[:] = clean_image
            cv2.imshow(window_title, image)
            points.clear()

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, 800, 800)
    cv2.imshow(window_title, image)
    cv2.setMouseCallback(window_title, on_click)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) != 4:
        raise RuntimeError("Make sure you select 4 corners!")

    return points


def find_template_corners_automatically(
    image: np.typing.NDArray[np.uint8],
) -> list[Point]:
    template_contour = find_largest_contour(image)
    corners = cv2.approxPolyDP(
        template_contour, 0.01 * cv2.arcLength(template_contour, True), True
    )

    if len(corners) != 4:
        raise CouldntFindCorners()

    return [(corner[0][0], corner[0][1]) for corner in corners]


def find_approximate_template_corners(
    image: np.typing.NDArray[np.uint8],
) -> list[Point]:
    try:
        corners = find_template_corners_automatically(image)
    except CouldntFindCorners:
        print("prompting")
        corners = prompt_user_for_template_corners(image)

    return corners


def find_relevant_x_and_ys_to_check(image: np.typing.NDArray[np.uint8]) -> Rectangle:
    corners = find_approximate_template_corners(image)
    corners_sorted_by_x = sorted(corners, key=lambda corner: corner[0])
    corners_sorted_by_y = sorted(corners, key=lambda corner: corner[1])
    # we add/subtract SAFETY_PIXELS in order to avoid removing
    # pixels by accident in our calculations
    min_x = corners_sorted_by_x[1][0] - SAFETY_PIXELS
    max_x = corners_sorted_by_x[2][0] + SAFETY_PIXELS
    min_y = corners_sorted_by_y[1][1] + SAFETY_PIXELS
    max_y = corners_sorted_by_y[2][1] - SAFETY_PIXELS
    return Rectangle(min_x, max_x, min_y, max_y)


def find_two_peaks(profile, distance):
    peaks, _ = scipy.signal.find_peaks(profile, distance=distance)
    if not len(peaks) == 2:
        raise InvalidNumberOfPeaks(
            f"found different number than 2 peaks (found {len(peaks)})"
        )
    return peaks


def get_improved_peak_location(
    profile: np.typing.NDArray[np.uint8], peak_location: int
) -> float:
    first_0_after_peak = profile[peak_location:].argmin(0)
    first_0_before_peak = profile[peak_location::-1].argmin(0)
    profile = profile[
        peak_location - first_0_before_peak : peak_location + first_0_after_peak + 1
    ]
    # we could also fit a gaussian or many other functions
    ((a, b, _), _) = scipy.optimize.curve_fit(parabola, range(len(profile)), profile)
    theoretical_relative_peak_location = -b / (2 * a)
    return peak_location - first_0_before_peak + theoretical_relative_peak_location


def parabola(x, a, b, c):
    return a * x * x + b * x + c


def find_distance(image: np.typing.NDArray[np.uint8], angle: float) -> float:
    rotated = rotate_image(image, angle)
    relevant_x_and_ys = find_relevant_x_and_ys_to_check(rotated)

    width = relevant_x_and_ys.max_x - relevant_x_and_ys.min_x

    distances = []
    for y in range(relevant_x_and_ys.min_y, relevant_x_and_ys.max_y + 1):
        profile = rotated[y, relevant_x_and_ys.min_x : relevant_x_and_ys.max_x + 1]
        peaks = find_two_peaks(profile, width / 2)
        first_peak_location = get_improved_peak_location(profile, peaks[0])
        second_peak_location = get_improved_peak_location(profile, peaks[1])
        distances.append(second_peak_location - first_peak_location)

    return np.mean(distances)  # type: ignore


def find_rectangle_inside_template(template_corners: list[Point]) -> Rectangle:
    corners_x_sorted = sorted(corner[0] for corner in template_corners)
    corners_y_sorted = sorted(corner[1] for corner in template_corners)

    min_x = corners_x_sorted[1] + SAFETY_PIXELS
    max_x = corners_x_sorted[2] - SAFETY_PIXELS
    min_y = corners_y_sorted[1] + SAFETY_PIXELS
    max_y = corners_y_sorted[2] - SAFETY_PIXELS

    return Rectangle(min_x, max_x, min_y, max_y)


def find_rectangle_containing_template(template_corners: list[Point]) -> Rectangle:
    min_x = min(corner[0] for corner in template_corners)
    max_x = max(corner[0] for corner in template_corners)
    min_y = min(corner[1] for corner in template_corners)
    max_y = max(corner[1] for corner in template_corners)
    return Rectangle(
        min_x - SAFETY_PIXELS,
        max_x + SAFETY_PIXELS,
        min_y - SAFETY_PIXELS,
        max_y + SAFETY_PIXELS,
    )


def sort_corners(points: list[Point]) -> list[Point]:
    midpoint = (np.mean([x for (x, y) in points]), np.mean([y for (x, y) in points]))
    return sorted(
        points,
        key=lambda point: np.atan2(point[1] - midpoint[1], point[0] - midpoint[0]),
    )


def determine_approximate_template_side_1_angle(
    sorted_template_corners: list[Point],
) -> float:
    point_1 = sorted_template_corners[0]
    point_2 = sorted_template_corners[1]

    return np.atan2(
        point_2[1] - point_1[1],
        point_2[0] - point_1[0],
    )


def determine_approximate_template_side_2_angle(
    sorted_template_corners: list[Point],
) -> float:
    point_1 = sorted_template_corners[1]
    point_2 = sorted_template_corners[2]

    return np.atan2(
        point_2[1] - point_1[1],
        point_2[0] - point_1[0],
    )


def find_distance_and_optimize_angle(
    image: np.typing.NDArray[np.uint8],
    starting_angle: float,
    angle_increment: float = np.radians(0.5),
) -> tuple[float, float]:
    starting_distance = find_distance(image, starting_angle)
    print(f"starting distance: {starting_distance}")

    try:
        distance_to_the_right = find_distance(image, starting_angle + angle_increment)
    except InvalidNumberOfPeaks:
        distance_to_the_right = math.inf

    try:
        distance_to_the_left = find_distance(image, starting_angle - angle_increment)
    except InvalidNumberOfPeaks:
        distance_to_the_left = math.inf

    if distance_to_the_right < starting_distance:
        direction = +1
        new_angle = starting_angle + angle_increment
        new_distance = distance_to_the_right
    elif distance_to_the_left < starting_distance:
        direction = -1
        new_distance = distance_to_the_left
        new_angle = starting_angle - angle_increment
    else:
        return starting_distance, starting_angle

    last_angle = starting_angle
    last_distance = starting_distance
    while new_distance < last_distance:
        last_distance = new_distance
        print(f"new distance: {last_distance}")
        last_angle = new_angle
        new_angle = last_angle + direction * angle_increment
        new_distance = find_distance(image, new_angle)

    print(
        f"optimization of angle moved us by {np.degrees(last_angle - starting_angle)} degrees"
    )
    print(f"final distance: {last_distance}")
    return last_distance, last_angle


def find_template(image: np.typing.NDArray) -> Template:
    no_bg_image = reduce_image_background(image)
    blurred_image = apply_gaussian_blur(no_bg_image, GAUSSIAN_BLUR_KERNEL_SIZE)

    gradients = find_gradients(blurred_image)
    # TODO consider automatic thresholding here (for example otsu thresholding)
    _, image = cv2.threshold(
        gradients.magnitudes.astype(np.uint8),
        20,
        255,
        cv2.THRESH_TOZERO,
    )

    template_corners = find_approximate_template_corners(image)
    rectangle_with_template = find_rectangle_containing_template(template_corners)
    # remove everything from the picture that is clearly not inside the template
    image[0 : rectangle_with_template.min_y, :] = 0
    image[rectangle_with_template.max_y + 1 :, :] = 0
    image[:, 0 : rectangle_with_template.min_x] = 0
    image[:, rectangle_with_template.max_x + 1 : image.shape[1]] = 0

    rectangle_inside_template = find_rectangle_inside_template(template_corners)
    # remove everything from the picture that is inside the template and doesn't include its boundaries
    image[
        rectangle_inside_template.min_y : rectangle_inside_template.max_y + 1,
        rectangle_inside_template.min_x : rectangle_inside_template.max_x + 1,
    ] = 0

    sorted_template_corners = sort_corners(template_corners)
    approximate_template_side_1_angle = determine_approximate_template_side_1_angle(
        sorted_template_corners
    )
    approximate_template_side_2_angle = determine_approximate_template_side_2_angle(
        sorted_template_corners
    )

    distance1, angle1 = find_distance_and_optimize_angle(
        image,
        # we need to add np.pi / 2 for the rotation to work properly
        approximate_template_side_1_angle + (np.pi / 2),
        angle_increment=np.radians(ANGLE_INCREMENT),
    )
    distance2, angle2 = find_distance_and_optimize_angle(
        image,
        approximate_template_side_2_angle + (np.pi / 2),
        angle_increment=np.radians(ANGLE_INCREMENT),
    )

    return Template(angle2 - angle1, distance1, distance2)


@click.command()
@click.option("--image-path", type=click.Path(), required=True)
def main(image_path: str) -> None:
    image = load_image(image_path)
    template = find_template(image)

    # TODO When our code will be better able to extract the grid from the image,
    #      we want to receive the path to two images - one of a relaxed image that we know
    #      the spacing of, and one of a stretched image, and we want to compute the spacing
    #      in the stretched image using these parameters. However, the program isn't reliable
    #      enough yet to be able to used, so for now we only print the properties of the
    #      template in the image passed to us.
    print(template)


if __name__ == "__main__":
    main()
