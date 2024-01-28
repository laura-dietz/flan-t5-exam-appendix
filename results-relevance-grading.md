
# Baselines: Relevance-judgment Prompts

Below detailed results of baseline prompts that directly ask an LLM to grade the relevance of a passage. 

| spearman | dl19 | dl20 |
| :-- | :-- | :-- |
| Fag | spearman        0.961     | spearman        0.951                                                    |
| Fag\_few | spearman        0.910           | spearman        0.909                                                    |
| Sun | spearman        0.964       | spearman        0.965                                                    |
| Sun\_few | spearman        0.741   | spearman        0.398                                                    |
| Thomas | spearman        0.855     | spearman        0.954                                                    |

| kendall | dl19 | dl20 |
| :-- | :-- | :-- |
| Fag | kendall 0.845      | kendall 0.848                                                    |
| Fag\_few | kendall 0.780      | kendall 0.766                                                    |
| Sun | kendall 0.847      | kendall 0.878                                                    |
| Sun\_few | kendall 0.546   |  kendall 0.277                                                    |
| Thomas | kendall 0.666         | kendall 0.841                                                    |

| kappa | dl19 | dl20 |
| :-- | :-- | :-- |
| Fag | Kappa 0.44 | Kappa 0.29 |
| Fag\_few | Kappa 0.26 | Kappa 0.17 |
| Sun | Kappa 0.39 | Kappa 0.24 |
| Sun\_few | Kappa -0.035 | Kappa -0.013 |
| Thomas | Kappa 0.084 | Kappa 0.078 |

We include these results as an upper-bound baseline.  These approaches directly mimick the relevance judgment process, but as a result it is difficult to integrate a human verifier into the loop --- while  avoiding human bias or costly manual judgments.

