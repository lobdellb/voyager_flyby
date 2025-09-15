
import cv2
import numpy as np



def find_circle_center_parametrized(
        img,
        blur_width,
        method: int,
        dp: float,
        minDist:float,
        param1:float = 100,
        param2:float = 100,
        minRadius:int = 0,
        maxRadius:int = 0 
    ):
    
    # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_blur = cv2.medianBlur(img, blur_width)

    # img_blur = img
    
    # circles = cv2.HoughCircles(
    #     img_blur, 
    #     cv2.HOUGH_GRADIENT, 
    #     dp=1, # was 1.2 
    #     minDist=100,
    #     param1=100, 
    #     param2=40, # was 30 
    #     minRadius=30, 
    #     maxRadius=2000
    # )

    circles = cv2.HoughCircles(
        img_blur, 
        method, 
        dp=dp, # was 1.2 
        minDist=minDist,
        param1=param1, 
        param2=param2,
        minRadius=minRadius, 
        maxRadius=maxRadius
    )

    return circles


def find_circle_center(img):


    # if circles is not None:
    #     circles = np.uint16(np.around(circles))

    #     x, y, r = circles[0][0]
            
    #     return int(x), int(y), int(r)

    # else:
    #     return 500,500,5

    return find_circle_center_parametrized(
        img,
        blur_width=5,
        method=cv2.HOUGH_GRADIENT_ALT,
        minDist=100,
        param1=24,
        param2=0.8,
        minRadius=30,
        maxRadius=2000 
    )





def center_object_in_larger_image(image, x, y, target_size=2000):
    """
    Center the object located at (x, y) in the input image into the middle
    of a larger (target_size x target_size) black image.
    
    Args:
        image: np.array of shape (H, W) or (H, W, C)
        x: x-coordinate of the object in the original image
        y: y-coordinate of the object in the original image
        target_size: size of the output image (default: 2000)

    Returns:
        new_image: np.array of shape (target_size, target_size) or (target_size, target_size, C)
    """

    # print( f"dtype {image.dtype}")
    
    H, W = image.shape[:3]
    C = 1 # if image.ndim == 2 else image.shape[2]

    # Create output black image
    if C == 1 and image.ndim == 2:
        new_image = np.zeros((target_size, target_size), dtype=image.dtype)
    else:
        new_image = np.zeros((target_size, target_size, C), dtype=image.dtype)

    # Calculate where to paste the original image

    # print( type( target_size ) )
    
    center_target = target_size // 2

    # print( type( center_target ) )
    
    offset_y = center_target - y
    offset_x = center_target - x

    # print( type( offset_y ) )c

    # print( f"offset_y {offset_y}")
    # print( f"offset_x {offset_x}")

    # Determine paste boundaries
    y_start_new = max(0, offset_y)
    x_start_new = max(0, offset_x)
    y_end_new = min(target_size, offset_y + H)
    x_end_new = min(target_size, offset_x + W)

    # print( f"y_start_new {y_start_new}")
    # print( f"x_start_new {x_start_new}")
    # print( f"y_end_new {y_end_new}")
    # print( f"x_end_new {x_end_new}")
    
    y_start_old = max(0, -offset_y)
    x_start_old = max(0, -offset_x)

    # print( f"----- {offset_y} , {-offset_y},  {max(0,-offset_y)}, {type(offset_y)}")
    
    y_end_old = y_start_old + (y_end_new - y_start_new)
    x_end_old = x_start_old + (x_end_new - x_start_new)

    # print( f"y_start_old {y_start_old}")
    # print( f"x_start_old {x_start_old}")
    # print( f"y_end_old {y_end_old}")
    # print( f"x_end_old {x_end_old}")
    
    # Copy the overlapping region
    if C == 1 and image.ndim == 2:
        new_image[y_start_new:y_end_new, x_start_new:x_end_new] = \
            image[y_start_old:y_end_old, x_start_old:x_end_old]
    else:
        new_image[y_start_new:y_end_new, x_start_new:x_end_new, :] = \
            image[y_start_old:y_end_old, x_start_old:x_end_old, :]

    return new_image


def normalize_clip(arr: np.ndarray, in_min: float, in_max: float) -> np.ndarray:
    """
    Normalize a numpy array such that in_min maps to 0 and in_max maps to 1.
    Values below in_min are clipped to 0, and values above in_max are clipped to 1.

    Parameters:
    - arr (np.ndarray): Input array of any shape.
    - in_min (float): Minimum input value to map to 0.
    - in_max (float): Maximum input value to map to 1.

    Returns:
    - np.ndarray: Normalized and clipped array with values in [0, 1].
    """
    if in_max == in_min:
        raise ValueError("in_max and in_min must be different to avoid division by zero.")

    # print( in_max )
    # print( in_min )
    # print( in_max - in_min )

    in_max = float( in_max )
    in_min = float( in_min )
    
    # Normalize
    norm = (arr - in_min) / (in_max - in_min)

    # Clip
    return np.clip(norm, 0.0, 1.0)

def scale_image(img: np.ndarray) -> np.ndarray:
    """
    Scale a (N, M, 1) int16 image to a float32 image with values in [0, 1].
    The dark background is preserved, and Saturn is enhanced for contrast.
    """
    # Remove singleton dimension
    # img = img.squeeze()

    # Convert to float for processing
    img = img.astype(np.float32)

    # Estimate low and high percentiles to ignore background & saturation
    low = np.percentile(img, 1)
    high = np.percentile(img, 99)

    # Prevent division by zero
    if high - low < 1e-5:
        return np.zeros_like(img)

    # Clip and scale
    img_scaled = np.clip((img - low) / (high - low), 0, 1)

    # Add back the singleton channel dimension
    return img_scaled[..., np.newaxis]

def norm_and_save_grey_image( image_arr, out_jpg_fn ):
    
    # normed_arr = normalize_clip( image_arr, in_min=image_arr.min() , in_max=image_arr.max() )
    normed_arr = scale_image( image_arr )

    dim1 = normed_arr.shape[1]
    dim2 = normed_arr.shape[2]

    plt.imsave( out_jpg_fn,
        np.stack([normed_arr]*3, axis=-1).reshape(dim1,dim2,3),
        cmap="grey"
    )

