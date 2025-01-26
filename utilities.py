import cv2
import numpy as np
import os


def draw_bboxes_and_lines(img, detection_results, grouped_lines,  output_dir="test_img_boxes", image_name="output_image.png"):
    """
    Function to visualize bounding boxes and grouped lines on the image using OpenCV,
    and save the result as a new image in the specified output directory.
    
    Args:
        img (np.array): Input image as a numpy array (BGR format for OpenCV).
        detection_results (dict): Detection results containing 'boxes' for individual bounding boxes.
        grouped_lines (dict): Dictionary containing grouped bounding boxes by lines.
        output_dir (str): Directory where the output image will be saved.
        image_name (str): Name of the image file to be saved.
    
    Returns:
        None: Saves the image with bounding boxes and grouped lines drawn.
    """
    # Ensure the output directory exists, if not create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Color settings (BGR format for OpenCV)
    bbox_color = (0, 255, 255)  # Yellow for individual bounding boxes
    line_color = (255, 0, 0)    # Blue for grouped lines
    linewidth = 2

    # Draw individual bounding boxes (before grouping)
    for bbox in detection_results['boxes']:
        x_min = min(bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0])
        y_min = min(bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1])
        x_max = max(bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0])
        y_max = max(bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1])
        
        # Draw a rectangle around the detected box (x_min, y_min) and (x_max, y_max)
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), bbox_color, linewidth)

    # Draw grouped lines (after grouping bounding boxes into lines)
    for bbox in grouped_lines['boxes']:
        x_min = min(bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0])
        y_min = min(bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1])
        x_max = max(bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0])
        y_max = max(bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1])
        
        # Draw a rectangle for each bounding box in the group (line)
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), line_color, linewidth)
    
    # Construct the full path for saving the image
    output_path = os.path.join(output_dir, image_name)
    
    # Save the result to the specified file path
    cv2.imwrite(output_path, img)

    print(f"Image saved to {output_path}")

def group_results_into_lines(detection_results, y_tolerance=10, x_tolerance=20):
    """
    Groups bounding boxes (bboxes) into lines based on their Y-axis positions and X-axis proximity.

    Args:
        detection_results (dict): OCR detection results containing "boxes".
        y_tolerance (int): Maximum allowed deviation on the Y-axis to consider boxes as part of the same line.
        x_tolerance (int): Maximum allowed horizontal space between bounding boxes in the same line.

    Returns:
        dict: A dictionary with grouped bounding boxes, where each group corresponds to a single line.
    """
    bboxes = []
    for polygon in detection_results["boxes"]:
        x_coords = polygon[:, 0]
        y_coords = polygon[:, 1]
        x1 = min(x_coords)
        y1 = min(y_coords)
        x2 = max(x_coords)
        y2 = max(y_coords)
        bboxes.append([x1, y1, x2, y2])

    # Sort the bounding boxes by their y1 coordinate (top-left corner Y-axis)
    bboxes = sorted(bboxes, key=lambda box: box[1])

    # Group boxes into lines
    grouped_lines = {"boxes": []}
    current_line = []
    x1_min, y1_min, x2_max, y2_max = bboxes[0]

    for box in bboxes:
        bx1, by1, bx2, by2 = box

        # Check Y-axis overlap or proximity
        if abs(by1 - y1_min) <= y_tolerance or \
        abs(by2 - y2_max) <= y_tolerance or \
        (by1 >= y1_min and by2 <= y2_max):
            # Check X-axis continuity
            if bx1 - x2_max <= x_tolerance:
                # Merge the box into the current line
                x1_min = min(x1_min, bx1)
                y1_min = min(y1_min, by1)
                x2_max = max(x2_max, bx2)
                y2_max = max(y2_max, by2)
                current_line.append(box)
            else:
                # Start a new line
                grouped_lines["boxes"].append(
                    np.array([[x1_min, y1_min], [x2_max, y1_min], [x2_max, y2_max], [x1_min, y2_max]])
                )
                x1_min, y1_min, x2_max, y2_max = bx1, by1, bx2, by2
                current_line = [box]
        else:
            # Start a new line when Y-axis proximity fails
            grouped_lines["boxes"].append(
                np.array([[x1_min, y1_min], [x2_max, y1_min], [x2_max, y2_max], [x1_min, y2_max]])
            )
            x1_min, y1_min, x2_max, y2_max = bx1, by1, bx2, by2
            current_line = [box]

    # Add the last line
    if current_line:
        grouped_lines["boxes"].append(
            np.array([[x1_min, y1_min], [x2_max, y1_min], [x2_max, y2_max], [x1_min, y2_max]])
        )

    return grouped_lines