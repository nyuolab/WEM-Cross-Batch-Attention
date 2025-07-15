# WEM-Cross-Batch-Attention

WEM (Whole Exome Model) is a model designed to handle the entire exome in context at once. It does this through a cross batch attention mechanism, which will pass information in between the "batches" in a run. While not batches in the sense that they represent completely independent input to the model, the name comes from the idea that with a simple transpose of the input matrix we can run self attention on a dimension that will pass information between multiple documents in the same forward pass. 

We are applying this to whole exomes by treating the Genes as the batch dimension and sequeneces for those genes as the sequence dimension. We hope that this architecture will allow us to perform tasks with very large context lengths


