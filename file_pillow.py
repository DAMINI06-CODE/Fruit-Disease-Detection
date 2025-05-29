from PIL import Image

print("Pillow is installed and working correctly.")

try:
    img = Image.open(r"C:\Users\DAMINI\Desktop\CG PROJECT\fruit-disease-detection\wolf.jpg")
    img.show()
    print("Image opened successfully.")
except Exception as e:
    print(f"An error occurred: {e}")
