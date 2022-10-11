---
layout: post
title: Finding Needles in a Paper Haystack
tags: [nlp]
---

#### Quick links:
- For code: [![Open In Colab]({{site.baseurl}}/assets/images/colab-badge.svg)](https://colab.research.google.com/github/)
- For the output visualization: [link]({{site.baseurl}}/assets/images/documentation/research-clustering/vis.html)

<hr/>

![clusters]({{site.baseurl}}/assets/images/documentation/research-clustering/clusters.png)

Given the deluge of research papers in AI, one often needs to (re)orient themselves to keep up with the latest and greatest—not to mention the “state-of-the-art”— developments, all the while exploring new and exciting research avenues.

But what if we could have a bird’s-eye view of conference papers, for example? With modern language models now able to fully parse and capture *contextual semantics* from text, we can finally resolve this information density problem and break away from classic, sequentially scroll-through-a-table for exploration. Think of it more as *visual* literature review.

Neural Topic Modeling (NTM) approaches such as [Top2Vec](https://github.com/ddangelov/Top2Vec) [1], CombinedTM [2] and [BERTopic](https://github.com/MaartenGr/BERTopic) [3]—to name a few—have been gaining traction recently [4], and we could use one such approach to cluster documents in an unsupervised manner with minimal preprocessing.

For this experiment, I decided to go with papers [accepted](https://proceedings.neurips.cc/paper/2021) at NeurIPS (2021). Here’s a preview of what the dataset (scraped from the site) looks like below:

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>title</th>
      <th>authors</th>
      <th>conference</th>
      <th>year</th>
      <th>abstract</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Beyond Value-Function Gaps: Improved Instance-...</td>
      <td>Christoph Dann, Teodor Vanislavov Marinov, Meh...</td>
      <td>neurips</td>
      <td>2021</td>
      <td>We provide improved gap-dependent regret bound...</td>
    </tr>
    <tr>
      <td>Learning One Representation to Optimize All Re...</td>
      <td>Ahmed Touati, Yann Ollivier</td>
      <td>neurips</td>
      <td>2021</td>
      <td>We introduce the forward-backward (FB) represe...</td>
    </tr>
    <tr>
      <td>Matrix factorisation and the interpretation of...</td>
      <td>Nick Whiteley, Annie Gray, Patrick Rubin-Delanchy</td>
      <td>neurips</td>
      <td>2021</td>
      <td>Given a graph or similarity matrix, we conside...</td>
    </tr>
    <tr>
      <td>UniDoc: Unified Pretraining Framework for Docu...</td>
      <td>Jiuxiang Gu, Jason Kuen, Vlad I Morariu, Hando...</td>
      <td>neurips</td>
      <td>2021</td>
      <td>Document intelligence automates the extraction...</td>
    </tr>
    <tr>
      <td>Finding Discriminative Filters for Specific De...</td>
      <td>Liangbin Xie, Xintao Wang, Chao Dong, Zhongang...</td>
      <td>neurips</td>
      <td>2021</td>
      <td>Recent blind super-resolution (SR) methods typ...</td>
    </tr>
  </tbody>
</table>
<p>2334 rows × 5 columns</p>


So, let's briefly define our goals below:
- We need to obtain a relevant dataset of papers and concatenate their `title` +  `abstract`, which will serve as our corpus (of size `N`)
- For paper discovery (à la Top2Vec and BERTopic):
    - Use a pre-trained language model (for faster inference: MiniLM or `all-MiniLM-L6-v2` with SentenceTransformers [5]) to embed documents into a `N x M` dimensional vector
    - Use UMAP [6] to obtain a low-dimensional projection, and
    - HDSCAN [7] for clustering

- Find salient clusters
- Identify interesting research papers/directions

Here's a schematic of the same:

![clustering schematic]({{site.baseurl}}/assets/images/documentation/research-clustering/clustering-schematic.png)

> After performing the steps above, we end up with the interactive 2D clusters [here]({{site.baseurl}}/assets/images/documentation/research-clustering/vis.html).

<!-- <iframe
  src="{{site.baseurl}}/assets/images/documentation/research-clustering/vis.html"
  style="width:100%; height:500px; display: block"
></iframe> -->

Some (personal) takeways:

- Reinforcement learning (RL) dominates a lot of the space.
- Causal Treatment Effects has a notable imprint (quite surprising and interesting).
- Language models (including topic modeling) occupy a decent area.
- Transformers have fully anchored itself across video, text, audio and images.
- There's a surprising amount of neuro-inspired and related deep learning papers (under `neural_brain_networks`).
- Sparsity and pruning in neural nets is still an active area of research (under `pruning_sparse_training`).

I encourage you to explore and find areas of *your* interest(s) among the clusters.

Note:

- A typical language model has the constraint of maximum sequence length (`256`, in case of MiniLM) that it can process, and beyond which the input sequence/text will be truncated.
- The clustering phase largely depends on two important hyperparameters: number of  neighbors (UMAP) and minimum cluster size (HDBSCAN). It is always advisable to try a few different values, or alternatively use topic coherence measures (NPMI, for example) for cross-validating the optimal values for the same.

<hr/>

### References

- [1] Angelov, D. (2020). Top2vec: Distributed representations of topics. arXiv preprint arXiv:2008.09470.
- [2] Bianchi, F., Terragni, S., & Hovy, D. (2021, August). Pre-training is a Hot Topic: Contextualized Document Embeddings Improve Topic Coherence. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers) (pp. 759-766).
- [3] Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv preprint arXiv:2203.05794.
- [4] Zhao, H., Phung, D., Huynh, V., Jin, Y., Du, L., & Buntine, W. (2021). Topic modelling meets deep neural networks: A survey. arXiv preprint arXiv:2103.00498.
- [5] Reimers, N., & Gurevych, I. (2019). Sentence-bert: Sentence embeddings using siamese bert-networks. arXiv preprint arXiv:1908.10084.
- [7] McInnes, L., Healy, J., & Astels, S. (2017). hdbscan: Hierarchical density based clustering. J. Open Source Softw., 2(11), 205.
- [6] McInnes, L., Healy, J., & Melville, J. (2018). Umap: Uniform manifold approximation and projection for dimension reduction. arXiv preprint arXiv:1802.03426.