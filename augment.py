import os
import random
from tkinter import Tk, filedialog, Button, Label, Canvas, PhotoImage, Checkbutton, IntVar
import uuid
from PIL import Image, ImageTk, ImageEnhance
import numpy as np

# Augmentation Functions
def rotate_image(image, angle=None):
    if angle is None:
        angle = random.randint(-30, 30)
    return image.rotate(angle)

def flip_image(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def flip_image_vertical(image):
    return image.transpose(Image.FLIP_TOP_BOTTOM)

def adjust_brightness(image, factor=None):
    if factor is None:
        factor = random.uniform(0.5, 1.5)
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def add_noise(image, noise_factor=0.1):
    img_array = np.array(image)
    noise = np.random.randn(*img_array.shape) * 255 * noise_factor
    noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

def random_crop_and_resize(image, crop_factor=0.8):
    width, height = image.size
    new_width = int(width * crop_factor)
    new_height = int(height * crop_factor)
    left = random.randint(0, width - new_width)
    top = random.randint(0, height - new_height)
    cropped_image = image.crop((left, top, left + new_width, top + new_height))
    return cropped_image.resize((width, height))

# Augmentation Pipeline
def augment_image(image, augmentations):
    for aug in augmentations:
        image = aug(image)
    return image

# GUI Functions
class AugmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Augmentation Tool")
        
        # Input Folder Path
        self.input_folder = None
        self.images = []
        
        # Mask Folder Path
        self.mask_folder = None
        self.masks = {}
        
        # Augmentation Options
        self.augmentation_options = {
            "Rotate": IntVar(),
            "Flip Horizontal": IntVar(),
            "Flip Vertical": IntVar(),
            "Adjust Brightness": IntVar(),
            "Add Noise": IntVar(),
            "Random Crop and Resize": IntVar()
        }
        
        # GUI Components
        self.label = Label(root, text="Select a folder of images to augment:")
        self.label.pack()
        
        self.select_button = Button(root, text="Select Image Folder", command=self.select_folder)
        self.select_button.pack()
        
        self.select_mask_button = Button(root, text="Select Mask Folder (Optional)", command=self.select_mask_folder)
        self.select_mask_button.pack()
        
        # Checkboxes for augmentations
        for aug_name, var in self.augmentation_options.items():
            Checkbutton(root, text=aug_name, variable=var).pack()
        
        self.augment_button = Button(root, text="Apply Augmentations", command=self.apply_augmentations, state="disabled")
        self.augment_button.pack()
        
        self.save_button = Button(root, text="Save Augmented Images", command=self.save_image, state="disabled")
        self.save_button.pack()
        
        self.canvas = Canvas(root, width=500, height=500)
        self.canvas.pack()
        
        self.status_label = Label(root, text="", fg="green")
        self.status_label.pack()
    
    def select_folder(self):
        self.input_folder = filedialog.askdirectory()
        if self.input_folder:
            self.images = [Image.open(os.path.join(self.input_folder, f)) for f in os.listdir(self.input_folder) if f.endswith(('.jpg', '.png'))]
            if self.images:
                self.display_image(self.images[0])
                self.augment_button.config(state="normal")
                self.status_label.config(text=f"Selected folder: {os.path.basename(self.input_folder)}", fg="blue")
    
    def select_mask_folder(self):
        self.mask_folder = filedialog.askdirectory()
        if self.mask_folder:
            self.masks = {os.path.splitext(f)[0].replace('_mask', ''): Image.open(os.path.join(self.mask_folder, f)) for f in os.listdir(self.mask_folder) if f.endswith(('.jpg', '.png')) and '_mask' in f}
            self.status_label.config(text=f"Selected mask folder: {os.path.basename(self.mask_folder)}", fg="blue")
    
    def apply_augmentations(self):
        if self.images:
            selected_augmentations = []
            if self.augmentation_options["Rotate"].get():
                selected_augmentations.append(rotate_image)
            if self.augmentation_options["Flip Horizontal"].get():
                selected_augmentations.append(flip_image)
            if self.augmentation_options["Flip Vertical"].get():
                selected_augmentations.append(flip_image_vertical)
            if self.augmentation_options["Adjust Brightness"].get():
                selected_augmentations.append(adjust_brightness)
            if self.augmentation_options["Add Noise"].get():
                selected_augmentations.append(add_noise)
            if self.augmentation_options["Random Crop and Resize"].get():
                selected_augmentations.append(random_crop_and_resize)
            
            self.augmented_images = {}
            self.augmented_masks = {}
            
            for img in self.images:
                img_name = os.path.basename(img.filename)
                img_base_name = os.path.splitext(img_name)[0]
                augmented_img = augment_image(img.copy(), selected_augmentations)
                self.augmented_images[img_base_name] = augmented_img
                
                if img_base_name in self.masks:
                    augmented_mask = augment_image(self.masks[img_base_name].copy(), selected_augmentations)
                    self.augmented_masks[img_base_name] = augmented_mask
            
            self.display_image(self.augmented_images[list(self.augmented_images.keys())[0]])
            self.save_button.config(state="normal")
            self.status_label.config(text="Augmentations applied to all images and corresponding masks!", fg="green")
    
    def save_image(self):
        if self.augmented_images:
            output_folder = filedialog.askdirectory()
            if output_folder:
                guid = uuid.uuid4()
                for img_name, img in self.augmented_images.items():
                    img.save(os.path.join(output_folder, f"{img_name}_ag{guid}.jpg"))
                
                for mask_name, mask in self.augmented_masks.items():
                    mask.save(os.path.join(output_folder, f"{mask_name}_ag{guid}_mask.jpg"))
                
                self.status_label.config(text=f"Images and masks saved in: {os.path.basename(output_folder)}", fg="green")
    
    def display_image(self, image):
        # Resize image for display in Canvas
        max_size = 500
        image.thumbnail((max_size, max_size))
        img_tk = ImageTk.PhotoImage(image)
        self.canvas.create_image(250, 250, anchor="center", image=img_tk)
        self.canvas.image = img_tk

# Main App Loop
if __name__ == "__main__":
    root = Tk()
    app = AugmentationApp(root)
    root.mainloop()
