from .llama_patcher import inject_surgery, patch_llama_model
from .surgery import SurgicalLlamaAttention, SurgeryLossRamp
from .surgery_trainer import SurgeryTrainer, TauAnnealingCallback
from .topology import DynamicTopologyRouter
from .qat import QATLinear, inject_qat, FakeQuantizeSTE

try:
    from .multimodal_injector import MultimodalEncoder, VisionProjection
except Exception:
    MultimodalEncoder = None
    VisionProjection = None

__all__ = [
    "inject_surgery",
    "patch_llama_model",
    "SurgicalLlamaAttention",
    "SurgeryLossRamp",
    "SurgeryTrainer",
    "TauAnnealingCallback",
    "DynamicTopologyRouter",
    "QATLinear",
    "inject_qat",
    "FakeQuantizeSTE",
    "MultimodalEncoder",
    "VisionProjection",
]
