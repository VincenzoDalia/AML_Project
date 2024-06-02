import torch


class ActivationShapingModule:
    def __init__(self, topK=False, binarize=True, topK_treshold=1, epsilon=0):
        self.topK = topK
        self.binarize = binarize
        self.topK_treshold = topK_treshold
        self.epsilon = epsilon

    def binarize_matrix(self, m):
        result = torch.ones_like(m) * self.epsilon
        result[m > 0] = 1.0
        return result

    def get_topK(self, A, k):
        _, top_indices = torch.topk(A.flatten(), k)
        # Creating the new A tensor with top K values
        A_topK = torch.zeros_like(A)
        A_topK.view(-1)[top_indices] = A.view(-1)[top_indices]

        return A_topK

    def topK_shape_or(self, A, M, t):
        if self.binarize:
          M = self.binarize_matrix(M)

        if self.topK_treshold == 1:
          return A * M 

        k = int(self.topK_treshold * A.numel())
        A_topK = self.get_topK(A, k)
        
        return A_topK * M



    def binarize_then_shape(self, A, M):
        M_binary = self.binarize_matrix(M)
        A_binary = self.binarize_matrix(A)
        return A_binary * M_binary

    def shape_activation(self, A, M):
        """
        Shape the output of a layer performing A * M

        Parameters:
        A : The output activation map of a layer.

        M: the matrix with which to perform the shaping.

        Returns:
        The result of A * M
        """
        if self.topK:
            return self.topK_shape(A, M, self.topK_treshold)

        if self.binarize:
            return self.binarize_then_shape(A, M)

        return A * M
