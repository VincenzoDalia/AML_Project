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
    
    def topK_shape(self, A, M, t):
        M_binary = self.binarize_matrix(M)

        k = int(self.topK_treshold * A.numel())

        _, top_indices = torch.topk(A.flatten(), k)
        # Creating the new A tensor with top K values
        A_topK = torch.zeros_like(A)
        A_topK.view(-1)[top_indices] = A.view(-1)[top_indices]

        return A_topK * M_binary

    def binarize_then_shape(self, A, M):
        M_binary = self.binarize_matrix(M)
        A_binary = self.binarize_matrix(A)
        return A_binary * M_binary

    def shape_activation(self, activation_map, M):
        """
        Shape the output of a layer performing A * M

        Parameters:
        activation_map : The output activation map of a layer.

        M: the matrix with which to perform the shaping.

        topK: if True keep only the topK elements of the activation map
        binarize: if True binarize both the activation map and M

        Returns:
        The result of A * M
        """

        if self.topK:
            return self.topK_shape(activation_map, M, self.topK_treshold)

        if self.binarize:
            return self.binarize_then_shape(activation_map, M)

        return activation_map * M
