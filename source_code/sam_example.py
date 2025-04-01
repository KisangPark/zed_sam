"""
Nano SAM

mask point label
    0) background point
    1) foreground point
    2) bounding box top left
    3) bounding box bottom right
"""

from nanosam.utils.predictor import Predictor

# define predictor model
predictor = Predictor(
    image_encoder="data/resnet18_image_encoder.engine",
    mask_decoder="data/mobile_sam_mask_decoder.engine"
)

# load image using PIL -> cv?
image = PIL.Image.open("dog.jpg")

# use predictor.set_image to load image to model
predictor.set_image(image)

# prediction -> return mask
mask, _, _ = predictor.predict(np.array([[x, y]]), np.array([1]))