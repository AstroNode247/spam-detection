from pydantic import BaseModel

class InferenceRequest(BaseModel):
        text: str


class InferenceResponse(BaseModel):
    input_text: str
    prediction: int
    label: str
