import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
from PIL import Image

class LimeExplainer_image:
    """
    LIME Image Explainer for Exact Library
    Works with Pytorch and Tensorflow
    Automatically handles image preprocessing and resizing.
    """

    def __init__(self, wrapped_model, num_samples=1000, target_size=(128, 128)):
        self.model = wrapped_model
        self.num_samples = num_samples  # Number of perturbed samples LIME generates
        self.target_size = target_size  # Target image size for model input
        self.explainer = lime_image.LimeImageExplainer()

    @staticmethod
    def _preprocess_image(image, target_size):
        """
        Preprocess image: convert to PIL, resize, and normalize.

        Args:
            image: PIL Image, numpy array, or file path (str)
            target_size: Tuple (height, width) to resize to

        Returns:
            numpy array: Normalized image in range [0, 1] with shape (H, W, C)
        """
        # Load image if it's a file path
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = (
                Image.fromarray(image.astype(np.uint8))
                if image.max() > 1
                else Image.fromarray((image * 255).astype(np.uint8))
            )

        # Resize to target size
        if isinstance(image, Image.Image):
            image = image.resize(target_size)

        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(image, dtype=np.float32) / 255.0

        return img_array

    def explain(self, image, top_labels=1):
        """
        Explain an image using LIME.

        Args:
            image: PIL Image, numpy array, or file path (str)
            top_labels: Number of top labels to explain

        Returns:
            LIME explanation object
        """
        # Preprocess image
        preprocessed_image = self._preprocess_image(image, self.target_size)

        def predict_fn(images):
            # Lime gives a list of images -> convert to numpy
            images = np.array(images)
            return self.model.predict_proba(images)

        explanation = self.explainer.explain_instance(
            image=preprocessed_image,
            classifier_fn=predict_fn,
            top_labels=top_labels,
            hide_color=0,
            num_samples=self.num_samples,
        )

        return explanation

    def get_visualization(
        self,
        explanation,
        label=None,
        positive_only=True,
        num_features=3,
        hide_rest=True,
    ):
        """
        Returns LIME visualization

        """

        if label is None:
            label = explanation.top_labels[0]

        image, mask = explanation.get_image_and_mask(
            label=label,
            positive_only=positive_only,
            num_features=num_features,  # How many features we should show
            hide_rest=hide_rest,  # Hide irrelevant regions
        )

        return mark_boundaries(image, mask)
    def overlay_heatmap(
        self,
        explanation,
        image=None,
        label=None,
        positive_only=True,
        num_features=10,
        hide_rest=True,
        show=False,
        save_png=False,
    ):
        """
        Generate LIME visualization overlayed on the original image.
        
        Args:
            explanation: LIME explanation object from explain()
            image: Original image (PIL Image, numpy array, or file path)
            label: Class label to explain (uses top label if None)
            positive_only: Show only positive contributions
            num_features: Number of features to highlight
            hide_rest: Hide regions not contributing to explanation
            show: Whether to display the visualization
            save_png: Whether to save the visualization to user_saves/
        
        Returns:
            numpy array: LIME visualization with boundaries overlayed
        """
        if label is None:
            label = explanation.top_labels[0]

        lime_viz, mask = explanation.get_image_and_mask(
            label=label,
            positive_only=positive_only,
            num_features=num_features,
            hide_rest=hide_rest,
        )

        # Get the visualization with boundaries
        visualization = mark_boundaries(lime_viz, mask)

        if show:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Show original image
            if image is not None:
                orig_img = self._preprocess_image(image, self.target_size)
                axes[0].imshow(orig_img)
                axes[0].set_title("Original Image")
            axes[0].axis("off")

            # Show LIME visualization
            axes[1].imshow(visualization)
            axes[1].set_title("LIME Explanation")
            axes[1].axis("off")

            plt.tight_layout()
            plt.show()

        if save_png:
            import os

            save_dir = "user_saves"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "lime_explanation.png")
            
            # Convert visualization to uint8 and save
            viz_uint8 = (visualization * 255).astype(np.uint8)
            Image.fromarray(viz_uint8).save(save_path)
            print(f"LIME visualization saved to {save_path}")

        return visualization