from dataclasses import dataclass
from typing import Callable

import taichi as ti

import genesis as gs

# we will use struct for DofsState and DofsInfo after Hugh adds array_struct feature to taichi
DofsState = ti.template()
DofsInfo = ti.template()


@ti.data_oriented
class RigidGlobalInfo:
    def __init__(self, n_dofs: int, n_entities: int, n_geoms: int, f_batch_shape: Callable):
        self.n_awake_dofs = ti.field(dtype=gs.ti_int, shape=f_batch_shape())
        self.awake_dofs = ti.field(dtype=gs.ti_int, shape=f_batch_shape(n_dofs))

        # for init_mass_mat()

        self.entity_max_dofs: ti.template() | None = None
        self.mass_mat: ti.template() | None = None
        self.mass_mat_L: ti.template() | None = None
        self.mass_mat_D_inv: ti.template() | None = None

        self._mass_mat_mask: ti.template() | None = None
        self.meaninertia: ti.template() | None = None

        self.mass_parent_mask: ti.template() | None = None
