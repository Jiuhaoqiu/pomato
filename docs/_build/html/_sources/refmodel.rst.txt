.. ref-model:

-------------------
The Model
-------------------


This chapter provides an overview on the model formulations used in pomatos optimization kernel.


.. math::
    \sum_{{n}_{z}} d^{el}_{n,t} &= \sum_{{co}_{z}} G_{co,t} - \sum_{{es}_{z}} D^{es}_{es,t} - \sum_{{ph}_{z}} D^{ph}_{ph,t} - \sum_{{d}_{z}} D^{d}_{d,t} \\
    &+ \sum_{zz} (EX_{zz,z,t} - EX_{z,zz,t}) \quad \forall t \in \text{T}, \quad z \in \text{Z}
    :label: enerbal

where :math:`z` is the index of zones in equation :eq:`enerbal`.