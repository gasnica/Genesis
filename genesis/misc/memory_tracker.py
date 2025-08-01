from dataclasses import is_dataclass, fields
import math

import taichi as ti

import genesis as gs
from genesis.engine.solvers.rigid.array_class import DataManager

class MemoryTracker:
    def __init__(self):
        self.fields = []

    @classmethod
    def add_field(cls, field, name):
        field.name = name
        _memory_tracker_instance.fields.append(field)

    @classmethod
    def print_memory_usage(cls):
        print("Taichi fields memory usage:")
        for f in _memory_tracker_instance.fields:
            print (f"{f.name:<20}: {math.prod(f.shape) * cls.size_of(f.dtype):>12,} bytes")

    @classmethod
    def size_of(cls, dtype):
        match dtype:
            case ti.f32:
                return 4
            case ti.f64:
                return 8
            case ti.i32:
                return 4
            case ti.i64:
                return 8
            case ti.u32:
                return 4
            case _:
                raise ValueError(f"Unsupported dtype: {dtype}")

    @classmethod
    def gather_memory_usage(cls, scene: gs.Scene):
        dm: DataManager = scene.sim.rigid_solver.data_manager
        all_arrays_and_sizes: list[tuple[str, int]] = []

        # process fields of DataManager that are of dataclass type
        for attr_name in dir(dm):
            if attr_name.startswith("__"):
                continue
            array_struct = getattr(dm, attr_name)
            # Check if it's a dataclass instance (for now, just check for __dataclass_fields__)
            if is_dataclass(array_struct) and not isinstance(array_struct, type)\
                or hasattr(array_struct, "_data_oriented"):
                # Placeholder for processing dataclass fields
                print (f"Processing array_struct: {attr_name}")
                for array_name in dir(array_struct):
                    # skip taichi attributes
                    if array_name.startswith("__") or array_name.startswith("_data_oriented"):
                        continue
                    array = getattr(array_struct, array_name)
                    # skip functions
                    if callable(array):
                        continue
                    size: int = math.prod(array.shape) * cls.size_of(array.dtype)

                    # handle ti.vectors and matrices
                    size_multiplier = 1
                    if hasattr(array, "n"):
                        size_multiplier *= array.n
                    if hasattr(array, "m"):
                        size_multiplier *= array.m

                    # print (f"Array {array_struct.__class__.__name__}.{array_name} size: {size:,} bytes")
                    # print (f"Array {attr_name}.{array_name} size: {size:,} bytes")
                    full_name = f"{attr_name}.{array_name}"
                    all_arrays_and_sizes.append((full_name, size * size_multiplier))

        # sort by size
        all_arrays_and_sizes.sort(key=lambda x: x[1], reverse=True)

        # print the top 10 arrays
        total_sizes = sum(size for _, size in all_arrays_and_sizes)
        for name, size in all_arrays_and_sizes[:50]:
            print (f"{name:<40} {size:>12,} bytes {100.0 *size/total_sizes:>8.2f}%")




_memory_tracker_instance = MemoryTracker()
