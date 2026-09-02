from pydantic import BaseModel, ConfigDict, Field


class PatientFeatures(BaseModel):
    """
    Input features for one patient, matching the heart-disease.csv columns
    used to train the model (see pandas-heart-disease for the full data dictionary).
    """

    age: int = Field(..., ge=0, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: int = Field(..., ge=0, description="Resting blood pressure (mm Hg)")
    chol: int = Field(..., ge=0, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)")
    restecg: int = Field(..., ge=0, le=2, description="Resting electrocardiographic results (0-2)")
    thalach: int = Field(..., ge=0, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina (1 = yes, 0 = no)")
    oldpeak: float = Field(..., ge=0, description="ST depression induced by exercise relative to rest")
    slope: int = Field(..., ge=0, le=2, description="Slope of the peak exercise ST segment (0-2)")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels colored by fluoroscopy (0-4)")
    thal: int = Field(..., ge=0, le=3, description="Thallium stress result (0-3)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 52,
                "sex": 1,
                "cp": 0,
                "trestbps": 125,
                "chol": 212,
                "fbs": 0,
                "restecg": 1,
                "thalach": 168,
                "exang": 0,
                "oldpeak": 1.0,
                "slope": 2,
                "ca": 2,
                "thal": 3,
            }
        }
    )


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="1 = heart disease predicted, 0 = no heart disease predicted")
    probability: float = Field(..., description="Predicted probability of heart disease (class 1)")
