from Generate_input_images import Generate_input_images
from Generate_ref_vector import Generate_ref_vector
from Generate_one_image import Generate_one_image
from Detect_wheat import Detect_Wheat
from pathlib import Path

# generate input images
# gen = Generate_input_images()
# gen.execute_generate()
# generate the reference vector
# gen_vect = Generate_ref_vector()
# gen_vect.execute_generate_ref()

# generate on image in case for specefic preprocessing
# one_image = Generate_one_image()
# display result
detect_wheat = Detect_Wheat(["2022"], (12, 1, 3, 4, 5, 6))
detect_wheat.display_result(5, [0.2004178, 0.6260978, 0.81834114, 0.83477926, 0.23763539, 0.1645725], "gist_stern", 1065, 523, 0.5)
