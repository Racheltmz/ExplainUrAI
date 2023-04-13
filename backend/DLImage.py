# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries

# Deep Learning Image Tasks
class DLImage():
    def __init__(self, model, img):
        self.model = model # H5 file
        self.img = img # File path

    # Explain the model with LIME and return the explanation, image, and mask
    def lime_explainer(self, input_img, model, show_positive=True, hide_rest=True):
        # Expand the dimension to 4d
        input_img = np.expand_dims(input_img, axis=0)
        # Explain image
        explainer = lime_image.LimeImageExplainer(random_state=1)
        explanation = explainer.explain_instance(input_img[0].astype('double'), model.predict, top_labels=5, hide_color=0, num_samples=1000)
        image, mask = explanation.get_image_and_mask(
                explanation.top_labels[0],
                positive_only=show_positive, 
                num_features=5,
                hide_rest=hide_rest)
        return explanation, image, mask
    
    # Display contributions in a heatmap
    def heatmap_explainer(self, explanation):
        # Select the same class explained on the figures above
        ind = explanation.top_labels[0]
        # Map each explanation weight to the corresponding superpixel
        dict_heatmap = dict(explanation.local_exp[ind])
        heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
        return heatmap

    def generate_report(self):
        # Call the lime_explainer function
        # With the original background
        explanation_bkgrd, image_bkgrd, mask_bkgrd = self.lime_explainer(self.img, self.model, show_positive=True, hide_rest=False)
        # Without the original background to focus on the important features
        explanation, image, mask = self.lime_explainer(self.img, self.model)
        # Display the original image, image with boundaries without background,
        # image with boundaries and the background, and a heatmap of the important features
        img_explanations = {
            "Original Image": self.img,
            "Model's Decision Boundary (background)": mark_boundaries(image_bkgrd.astype(np.uint8), mask_bkgrd),
            "Model's Decision Boundary": mark_boundaries(image.astype(np.uint8), mask),
            "Heatmap of important features": self.heatmap_explainer(explanation)
        }
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        for (title, img), ax in zip(img_explanations.items(), axes.ravel()):
            if title != list(img_explanations.keys())[-1]:
                ax.imshow(img)
            else:
                # Plot the heatmap
                heatmap_img = ax.imshow(img, cmap = 'Blues', vmin = -img.max(), vmax = img.max())
                plt.colorbar(heatmap_img)
            ax.set_title(title, size=14)
            ax.axis('off')
        plt.show()