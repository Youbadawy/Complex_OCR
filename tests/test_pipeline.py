import os
import sys
import pytest
import cv2
import numpy as np
from pathlib import Path

# Update Python path to include project root
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from Mammo_AI_Src.image_processing.preprocessor import ImagePreprocessor
from Mammo_AI_Src.image_processing.ocr_engine import OCREngine
from Mammo_AI_Src.ml_pipeline.risk_assessment import RiskAssessmentEngine, RiskFactors
from Mammo_AI_Src.data_processing.feature_extractor import FeatureExtractor

class TestMammographyPipeline:
    @pytest.fixture(scope="class")
    def setup(self):
        """Setup test resources."""
        # Create test image
        test_image = np.ones((1024, 1024), dtype=np.uint8) * 128
        # Add some text and features to test image
        cv2.putText(test_image, "BIRADS: 4", (100, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.circle(test_image, (512, 512), 50, (200, 200, 200), -1)
        
        # Save test image
        test_dir = Path("tests/test_data")
        test_dir.mkdir(parents=True, exist_ok=True)
        test_image_path = test_dir / "test_mammogram.png"
        cv2.imwrite(str(test_image_path), test_image)
        
        return {
            "test_image_path": test_image_path,
            "test_image": test_image
        }

    def test_image_preprocessor(self, setup):
        """Test image preprocessing."""
        preprocessor = ImagePreprocessor()
        image = cv2.imread(str(setup["test_image_path"]))
        processed_image = preprocessor.preprocess_image(image)
        
        assert processed_image is not None
        assert processed_image.shape == (1024, 1024)
        assert processed_image.dtype == np.float64
        
        # Test normalization
        assert np.min(processed_image) >= 0
        assert np.max(processed_image) <= 1

    def test_feature_extraction(self, setup):
        """Test feature extraction."""
        preprocessor = ImagePreprocessor()
        feature_extractor = FeatureExtractor()
        
        # Process image
        image = cv2.imread(str(setup["test_image_path"]))
        processed_image = preprocessor.preprocess_image(image)
        features = feature_extractor.extract_features(processed_image)
        
        assert features is not None
        assert features.density_score is not None
        assert isinstance(features.mass_characteristics, dict)
        assert isinstance(features.symmetry_score, float)

    def test_ocr_engine(self, setup):
        """Test OCR capabilities."""
        ocr_engine = OCREngine()
        image = cv2.imread(str(setup["test_image_path"]))
        ocr_results = ocr_engine.process_image(image)
        
        assert isinstance(ocr_results, dict)
        assert 'combined_text' in ocr_results
        assert 'individual_results' in ocr_results
        assert 'birads' in ocr_results['combined_text'].lower()

    def test_risk_assessment(self):
        """Test risk assessment engine."""
        risk_engine = RiskAssessmentEngine()
        
        # Test low risk case
        low_risk = RiskFactors(
            age=35,
            family_history=False,
            previous_biopsies=0,
            breast_density="scattered",
            birads_score=2
        )
        low_risk_result = risk_engine.calculate_risk_score(low_risk)
        
        assert low_risk_result['risk_category'] == 'low'
        assert low_risk_result['risk_score'] < 0.5
        
        # Test high risk case
        high_risk = RiskFactors(
            age=65,
            family_history=True,
            previous_biopsies=2,
            breast_density="dense",
            birads_score=4
        )
        high_risk_result = risk_engine.calculate_risk_score(high_risk)
        
        assert high_risk_result['risk_category'] in ['high', 'very_high']
        assert high_risk_result['risk_score'] > 0.5

    def test_end_to_end_pipeline(self, setup):
        """Test complete pipeline."""
        # Initialize components
        preprocessor = ImagePreprocessor()
        feature_extractor = FeatureExtractor()
        ocr_engine = OCREngine()
        risk_engine = RiskAssessmentEngine()
        
        # Process image
        image = cv2.imread(str(setup["test_image_path"]))
        
        # Step 1: Preprocess
        processed_image = preprocessor.preprocess_image(image)
        assert processed_image is not None
        
        # Step 2: Extract features
        features = feature_extractor.extract_features(processed_image)
        assert features is not None
        
        # Step 3: OCR
        ocr_results = ocr_engine.process_image(image)
        assert ocr_results is not None
        
        # Step 4: Risk Assessment
        risk_factors = RiskFactors(
            age=50,
            family_history=True,
            previous_biopsies=1,
            breast_density="heterogeneous",
            birads_score=4
        )
        risk_results = risk_engine.calculate_risk_score(risk_factors)
        assert risk_results is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
