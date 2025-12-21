"""
Image data loader for computer vision tasks.
Supports classification (supervised) and feature extraction (unsupervised).
"""
import numpy as np
import pandas as pd
import mlflow
from pathlib import Path
from typing import Tuple, Optional, Callable
from PIL import Image
from sklearn.model_selection import train_test_split

from .base_loader import BaseDataLoader


class ImageDataLoader(BaseDataLoader):
    """
    Data loader for image datasets.
    
    Supports two structures:
    
    1. Directory structure (supervised classification):
       data/
         train/
           class_a/
             img1.jpg
             img2.jpg
           class_b/
             img3.jpg
        
    2. CSV with paths (supervised or unsupervised):
       image_path,label
       data/img1.jpg,cat
       data/img2.jpg,dog
       
    3. Directory of images (unsupervised):
       data/
         img1.jpg
         img2.jpg
    
    Examples:
        # Classification from directory
        loader = ImageDataLoader(
            data_path="data/images",
            structure_type="directory",
            target_column=None  # Classes from folder names
        )
        
        # From CSV
        loader = ImageDataLoader(
            data_path="data/manifest.csv",
            structure_type="csv",
            target_column="label",
            image_column="path"
        )
        
        # Unsupervised (feature extraction)
        loader = ImageDataLoader(
            data_path="data/unlabeled",
            structure_type="directory",
            target_column=None
        )
    """
    
    def __init__(
        self,
        data_path: str,
        structure_type: str = "directory",
        target_column: Optional[str] = None,
        image_column: str = "image_path",
        test_size: float = 0.2,
        validation_size: float = 0.0,
        random_state: int = 42,
        image_size: Tuple[int, int] = (224, 224),
        transform: Optional[Callable] = None,
        valid_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    ):
        """
        Initialize image data loader.
        
        Args:
            data_path: Path to images directory or CSV file
            structure_type: 'directory' or 'csv'
            target_column: Target column (None for unsupervised or when using directory structure)
            image_column: Column name with image paths (for CSV mode)
            test_size: Test set fraction
            validation_size: Validation set fraction
            random_state: Random seed
            image_size: Target image size (height, width)
            transform: Optional transform function for images
            valid_extensions: Valid image file extensions
        """
        super().__init__(
            data_path=data_path,
            target_column=target_column,
            test_size=test_size,
            validation_size=validation_size,
            random_state=random_state
        )
        
        self.structure_type = structure_type
        self.image_column = image_column
        self.image_size = image_size
        self.transform = transform
        self.valid_extensions = valid_extensions
        
        # Build image manifest
        self.manifest = self._build_manifest()
        
        # Determine if supervised (has labels)
        self.is_supervised = 'label' in self.manifest.columns
    
    def _build_manifest(self) -> pd.DataFrame:
        """Build manifest DataFrame with image paths and labels."""
        if self.structure_type == "directory":
            return self._build_from_directory()
        elif self.structure_type == "csv":
            return self._build_from_csv()
        else:
            raise ValueError(f"Unknown structure_type: {self.structure_type}")
    
    def _build_from_directory(self) -> pd.DataFrame:
        """
        Build manifest from directory structure.
        
        If subdirectories exist, treat as classes (supervised).
        Otherwise, treat as unlabeled images (unsupervised).
        """
        data_path = Path(self.data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Directory not found: {data_path}")
        
        # Check if there are subdirectories (class folders)
        subdirs = [d for d in data_path.iterdir() if d.is_dir()]
        
        records = []
        
        if subdirs:
            # Supervised: subdirectories are classes
            print(f"Found {len(subdirs)} class folders")
            for class_dir in subdirs:
                class_name = class_dir.name
                for img_path in class_dir.rglob('*'):
                    if img_path.suffix.lower() in self.valid_extensions:
                        records.append({
                            'image_path': str(img_path),
                            'label': class_name
                        })
        else:
            # Unsupervised: flat directory of images
            print("No class folders found - treating as unsupervised")
            for img_path in data_path.rglob('*'):
                if img_path.suffix.lower() in self.valid_extensions:
                    records.append({
                        'image_path': str(img_path)
                    })
        
        if not records:
            raise ValueError(f"No images found in {data_path}")
        
        return pd.DataFrame(records)
    
    def _build_from_csv(self) -> pd.DataFrame:
        """Build manifest from CSV file."""
        df = pd.read_csv(self.data_path)
        
        if self.image_column not in df.columns:
            raise ValueError(
                f"Image column '{self.image_column}' not found in CSV.\n"
                f"Available: {df.columns.tolist()}"
            )
        
        # Rename to standardized column names
        manifest = df.rename(columns={self.image_column: 'image_path'})
        
        if self.target_column and self.target_column in df.columns:
            manifest = manifest.rename(columns={self.target_column: 'label'})
        
        # Validate image paths exist
        missing = []
        for _, row in manifest.iterrows():
            if not Path(row['image_path']).exists():
                missing.append(row['image_path'])
        
        if missing:
            print(f"⚠️  Warning: {len(missing)} image files not found")
            if len(missing) <= 5:
                for path in missing:
                    print(f"  - {path}")
        
        return manifest
    
    def load_and_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series], 
                                      Optional[pd.Series], Optional[pd.DataFrame], Optional[pd.Series]]:
        """
        Load and split image dataset.
        
        Returns:
            For supervised: (X_train, X_test, y_train, y_test, X_val, y_val)
                where X contains image paths as DataFrames
            For unsupervised: (X_train, X_test, None, None, X_val, None)
        """
        if self.is_supervised:
            return self._split_supervised(self.manifest)
        else:
            return self._split_unsupervised(self.manifest)
    
    def _split_supervised(self, df: pd.DataFrame) -> Tuple:
        """Split for supervised learning."""
        X = df[['image_path']]
        y = df['label']
        
        # Stratify by class
        stratify = y if self._is_classification(y) else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=stratify
        )
        
        # Optional validation split if validation_size > 0
        X_val, y_val = None, None
        if self.validation_size > 0:
            val_size = self.validation_size / (1 - self.test_size)
            stratify_val = y_train if self._is_classification(y_train) else None
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=val_size, random_state=self.random_state, stratify=stratify_val
            )
        
        return X_train, X_test, y_train, y_test, X_val, y_val
    
    def _split_unsupervised(self, df: pd.DataFrame) -> Tuple:
        """Split for unsupervised learning."""
        X = df[['image_path']]
        
        X_train, X_test = train_test_split(
            X, test_size=self.test_size, random_state=self.random_state
        )
        
        # Optional validation split if validation_size > 0
        X_val = None
        if self.validation_size > 0:
            val_size = self.validation_size / (1 - self.test_size)
            X_train, X_val = train_test_split(
                X_train, test_size=val_size, random_state=self.random_state
            )
        
        return X_train, X_test, None, None, X_val, None
    
    def load_images(self, image_paths: pd.DataFrame, as_array: bool = True) -> np.ndarray:
        """
        Load images from DataFrame of paths.
        
        Args:
            image_paths: DataFrame with 'image_path' column
            as_array: If True, return numpy array. If False, return list of PIL Images.
        
        Returns:
            Numpy array of shape (N, H, W, C) or list of PIL Images
        """
        images = []
        
        for path in image_paths['image_path']:
            try:
                img = Image.open(path).convert('RGB')
                img = img.resize(self.image_size)
                
                if self.transform:
                    img = self.transform(img)
                
                if as_array:
                    img = np.array(img)
                
                images.append(img)
                
            except Exception as e:
                print(f"⚠️  Failed to load {path}: {e}")
                # Use black image as placeholder
                if as_array:
                    images.append(np.zeros((*self.image_size, 3), dtype=np.uint8))
                else:
                    images.append(Image.new('RGB', self.image_size))
        
        if as_array:
            return np.stack(images)
        return images
    
    def log_to_mlflow(self, context: str = "training") -> mlflow.data.dataset.Dataset:
        """
        Log image dataset to MLflow with proper source information.
        
        Args:
            context: Context for the dataset
        
        Returns:
            MLflow Dataset object
        """
        # Create dataset from manifest
        dataset = mlflow.data.from_pandas(
            self.manifest,
            source=str(self.data_path),
            targets="label" if self.is_supervised else None,
            name=f"images_{self.data_path.stem}"
        )
        
        # Log to MLflow
        mlflow.log_input(dataset, context=context)
        
        # Log metadata
        mlflow.set_tag("data.loader_type", "image")
        mlflow.set_tag("data.structure_type", self.structure_type)
        mlflow.set_tag("data.image_size", f"{self.image_size[0]}x{self.image_size[1]}")
        mlflow.set_tag("data.num_images", len(self.manifest))
        mlflow.set_tag("data.source_type", f"image_{self.structure_type}")
        
        if self.is_supervised:
            mlflow.set_tag("data.num_classes", self.manifest['label'].nunique())
            mlflow.set_tag("data.classes", str(sorted(self.manifest['label'].unique().tolist())))
        
        # Log sample images
        sample_images = self.manifest.head(5)
        for idx, row in sample_images.iterrows():
            try:
                img = Image.open(row['image_path'])
                mlflow.log_image(img, f"sample_images/image_{idx}.png")
            except Exception as e:
                print(f"⚠️  Could not log sample image: {e}")
        
        return dataset
    
    def get_data_info(self) -> dict:
        """Get dataset metadata."""
        info = {
            "data.loader_type": "image",
            "data.structure_type": self.structure_type,
            "data.path": str(self.data_path),
            "data.num_images": len(self.manifest),
            "data.image_size": f"{self.image_size[0]}x{self.image_size[1]}",
            "data.task_type": "supervised" if self.is_supervised else "unsupervised",
            "data.test_size": self.test_size,
            "data.validation_size": self.validation_size,
            "data.random_state": self.random_state,
        }
        
        if self.is_supervised:
            info["data.target"] = "label"
            info["data.target_type"] = "classification"
            info["data.num_classes"] = self.manifest['label'].nunique()
            
            # Class distribution
            class_dist = self.manifest['label'].value_counts().to_dict()
            info["data.class_distribution"] = str(class_dist)
        
        return info
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        info = self.get_data_info()
        
        lines = [
            "="*60,
            "IMAGE DATASET SUMMARY",
            "="*60,
            f"Source: {self.data_path}",
            f"Structure: {self.structure_type}",
            f"Images: {info['data.num_images']:,}",
            f"Image Size: {info['data.image_size']}",
            "",
            f"Task: {info['data.task_type'].upper()}",
        ]
        
        if self.is_supervised:
            lines.extend([
                f"Classes: {info['data.num_classes']}",
                "",
                "Class Distribution:",
            ])
            
            class_dist = self.manifest['label'].value_counts()
            for class_name, count in class_dist.items():
                pct = (count / len(self.manifest)) * 100
                lines.append(f"  {class_name}: {count} ({pct:.1f}%)")
        
        lines.append("="*60)
        return "\n".join(lines)
