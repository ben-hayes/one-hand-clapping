import evotorch
import torch
from EvoCLAP.vsthost import VstiHost


# FitnessFunction.__init__(
#     vsti_host: VstiHost
#     clap_similarity: CLAPSimilarity
# )
# FitnessFunction.compute(
#     solutions: evotorch.SolutionBatch
# )

# CLAPSimilarity.__init__(
#     batch_size: int,
#     device: torch.Device,
# )
# CLAPSimilarity.compute(
#     audio: torch.Tensor,
#     target_embedding: torch.Tensor
# )
# CLAPSimilarity.get_text_embedding(
#     input: str
# )
# CLAPSimilarity.get_audio_embedding(
#     input: torch.Tensor
# )


class FitnessFunction:
   def __init__(self,
                vsti_host= VstiHost(),
                clap_similarity= CLAPSimilarity(), 
                ):
        self.vsti_host = vsti_host
        self.clap_similarity = clap_similarity
    
    def compute(self,
                solutions: evotorch.SolutionBatch):
        return self.clap_similarity.compute(solutions, self.vsti_host)
   

   
class CLAPSimilarity:
    def __init__(self,
                 batch_size: int,
                 device: torch.device):
        self.batch_size = batch_size
        self.device = device
        self.audio_embedding = None
        self.text_embedding = None

    def compute(self,
                audio: torch.Tensor,
                target_embedding: torch.Tensor):
        self.audio_embedding = self.get_audio_embedding(audio)
        self.text_embedding = target_embedding
        return self.audio_embedding @ self.text_embedding

    def get_text_embedding(self,
                          input: str):
        return torch.rand(1, 10, device=self.device)

    def get_audio_embedding(self,
                            input: torch.Tensor):
        return torch.rand(1, 10, device=self.device)



