import Formalization.Combinatorics.PrefixSparsity

/-!
# Combinatorial Prefix-Sharing and Sparsity on Trees

This module re-exports the exact combinatorial prefix-sharing and sparsity results
from `Formalization.Combinatorics.PrefixSparsity`.

## Main Re-exported Results
- `card_shared_prefix`: Exact cardinality of path pairs sharing an $r$-prefix in a $p$-ary tree of depth $d$.
- `total_pairs_card`: Total number of path pairs ($p^{2d}$).
- `fraction_eq_p_inv_r`: The active fraction is exactly $p^{-r}$.
- `sparsity_bound`: Exact sparsity fraction $1 - p^{-r}$.
- Concrete evaluations: `sparsity_p2_r1` (50%), `sparsity_p2_r3` (87.5%), `sparsity_p2_r6` (98.4375%).
-/

