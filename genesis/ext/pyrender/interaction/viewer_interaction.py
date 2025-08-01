from typing import TYPE_CHECKING, cast
from typing_extensions import override
from threading import Lock as threading_Lock

import numpy as np
import taichi as ti

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.entities.rigid_entity.rigid_entity import RigidEntity

from .aabb import AABB, OBB
from .mouse_spring import MouseSpring
from .ray import Plane, Ray, RayHit
from .vec3 import Pose, Quat, Vec3, Color
from .viewer_interaction_base import ViewerInteractionBase, EVENT_HANDLE_STATE, EVENT_HANDLED

from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
import genesis.utils.array_class as array_class
from genesis.utils.tools import Timer2

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity.rigid_geom import RigidGeom
    from genesis.engine.entities.rigid_entity.rigid_link import RigidLink
    from genesis.engine.scene import Scene
    from genesis.ext.pyrender.node import Node


@ti.dataclass
class tiRayHit:
    # point: ti.types.vector.f32 #ti.math.vec3
    # normal: ti.types.vector.f32 # ti.math.vec3
    position: ti.math.vec3
    normal: ti.math.vec3
    distance: ti.float32


@ti.data_oriented
class ViewerInteraction(ViewerInteractionBase):
    """Functionalities to be implemented:
    - mouse picking
    - mouse dragging
    """

    def __init__(self, 
        camera: 'Node', 
        scene: 'Scene', 
        viewport_size: tuple[int, int], 
        camera_yfov: float,
        log_events: bool = False,
        camera_fov: float = 60.0,
    ) -> None:
        super().__init__(log_events)
        self.camera: 'Node' = camera
        self.scene: 'Scene' = scene
        self.viewport_size: tuple[int, int] = viewport_size
        self.camera_yfov: float = camera_yfov

        self.tan_half_fov: float = np.tan(0.5 * self.camera_yfov)
        self.prev_mouse_pos: tuple[int, int] = (viewport_size[0] // 2, viewport_size[1] // 2)

        self.picked_link: RigidLink | None = None
        self.picked_point_in_local: Vec3 | None = None
        self.mouse_drag_plane: Plane | None = None
        self.prev_mouse_3d_pos: Vec3 | None = None

        self.mouse_spring: MouseSpring = MouseSpring()
        self.lock = threading_Lock()

        # todo:shorter alias ? ti.math.vec3.field?
        self.temp_raycast_out = ti.Vector.field(3, dtype=gs.ti_float, shape=())
        self.temp_rayhit_out = tiRayHit.field(shape=())

    @override
    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> EVENT_HANDLE_STATE:
        super().on_mouse_motion(x, y, dx, dy)
        self.prev_mouse_pos = (x, y)

    @override
    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> EVENT_HANDLE_STATE:
        super().on_mouse_drag(x, y, dx, dy, buttons, modifiers)
        self.prev_mouse_pos = (x, y)
        if self.picked_link:
            # actual processing done in update_on_sim_step()

            return EVENT_HANDLED

    @override
    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> EVENT_HANDLE_STATE:
        super().on_mouse_press(x, y, button, modifiers)
        if button == 1: # left mouse button
            ray_hit = self.raycast_against_entities(self.screen_position_to_ray(x, y))
            with self.lock:
                if ray_hit.geom:
                    self.picked_link = ray_hit.geom.link
                    assert self.picked_link is not None

                    temp_fwd = self.get_camera_forward()
                    temp_back = -temp_fwd

                    self.mouse_drag_plane = Plane(temp_back, ray_hit.position)
                    self.prev_mouse_3d_pos = ray_hit.position

                    pose: Pose = Pose.from_link(self.picked_link)
                    self.picked_point_in_local = pose.inverse_transform_point(ray_hit.position)

                    self.mouse_spring.attach(self.picked_link, ray_hit.position)

    @override
    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int) -> EVENT_HANDLE_STATE:
        super().on_mouse_release(x, y, button, modifiers)
        if button == 1: # left mouse button
            with self.lock:
                self.picked_link = None
                self.picked_point_in_local = None
                self.mouse_drag_plane = None
                self.prev_mouse_3d_pos = None

                self.mouse_spring.detach()

    @override
    def on_resize(self, width: int, height: int) -> EVENT_HANDLE_STATE:
        super().on_resize(width, height)
        self.viewport_size = (width, height)
        self.tan_half_fov = np.tan(0.5 * self.camera_yfov)

    @override
    def update_on_sim_step(self) -> None:
        Timer2.begin("interaction.update_on_sim_step")
        with self.lock:
            if self.picked_link:
                mouse_ray: Ray = self.screen_position_to_ray(*self.prev_mouse_pos)
                ray_hit: RayHit = self.mouse_drag_plane.raycast(mouse_ray)
                assert ray_hit.is_hit
                if ray_hit.is_hit:
                    new_mouse_3d_pos: Vec3 = ray_hit.position
                    delta_3d_pos: Vec3 = new_mouse_3d_pos - self.prev_mouse_3d_pos
                    self.prev_mouse_3d_pos = new_mouse_3d_pos

                    use_force: bool = True
                    if use_force:
                        # apply force
                        self.mouse_spring.apply_force(new_mouse_3d_pos, self.scene.sim.dt)
                    else:
                        #apply displacement
                        pos = Vec3.from_tensor(self.picked_link.entity.get_pos())
                        pos += delta_3d_pos
                        self.picked_link.entity.set_pos(pos.as_tensor())
        Timer2.end()

    @override
    def on_draw(self) -> None:
        super().on_draw()
        return
        Timer2.begin("interaction.on_draw")
        if self.scene._visualizer is not None and self.scene._visualizer.viewer_lock is not None:
            self.scene.clear_debug_objects()
            mouse_ray: Ray = self.screen_position_to_ray(*self.prev_mouse_pos)

            Timer2.begin("raycast scene")
            closest_hit = self.raycast_against_entities(mouse_ray)
            if not closest_hit.is_hit:
                closest_hit = self._raycast_against_ground_plane(mouse_ray)
            Timer2.end()

            with self.lock:
                if self.picked_link:
                    assert self.mouse_drag_plane is not None
                    assert self.picked_point_in_local is not None

                    # draw held point
                    pose: Pose = Pose.from_link(self.picked_link)
                    held_point: Vec3 = pose.transform_point(self.picked_point_in_local)
                    self.scene.draw_debug_sphere(held_point.v, 0.02, Color.red().tuple())

                    plane_hit: RayHit = self.mouse_drag_plane.raycast(mouse_ray)
                    if plane_hit.is_hit:
                        self.scene.draw_debug_sphere(plane_hit.position.v, 0.02, Color.red().tuple())
                        self.scene.draw_debug_line(held_point.v, plane_hit.position.v, color=Color.red().tuple())
                else:
                    if closest_hit.is_hit:
                        self.scene.draw_debug_sphere(closest_hit.position.v, 0.01, (0, 1, 0, 1))
                        self._draw_arrow(closest_hit.position, 0.25 * closest_hit.normal, (0, 1, 0, 1))
                    if closest_hit.geom:
                        self._draw_entity_unrotated_obb(closest_hit.geom)
                        self._raycast_geom(closest_hit.geom, mouse_ray)
        Timer2.end()


    def screen_position_to_ray(self, x: float, y: float) -> Ray:
        # convert screen position to ray
        if True:
            x = x - 0.5 * self.viewport_size[0]
            y = y - 0.5 * self.viewport_size[1]
            x = 2.0 * x / self.viewport_size[1] * self.tan_half_fov
            y = 2.0 * y / self.viewport_size[1] * self.tan_half_fov
        else:
            # alternative way
            projection_matrix = self.camera.camera.get_projection_matrix(*self.viewport_size)
            x = x - 0.5 * self.viewport_size[0]
            y = y - 0.5 * self.viewport_size[1]
            x = 2.0 * x / self.viewport_size[0] / projection_matrix[0, 0]
            y = 2.0 * y / self.viewport_size[1] / projection_matrix[1, 1]

        # Note: ignoring pixel aspect ratio

        mtx = self.camera.matrix
        position = Vec3.from_array(mtx[:3, 3])
        forward = Vec3.from_array(-mtx[:3, 2])
        right = Vec3.from_array(mtx[:3, 0])
        up = Vec3.from_array(mtx[:3, 1])

        direction = forward + right * x + up * y
        return Ray(position, direction)

    def get_camera_forward(self) -> Vec3:
        mtx = self.camera.matrix
        return Vec3.from_array(-mtx[:3, 2])

    def get_camera_ray(self) -> Ray:
        mtx = self.camera.matrix
        position = Vec3.from_array(mtx[:3, 3])
        forward = Vec3.from_array(-mtx[:3, 2])
        return Ray(position, forward)

    def _raycast_against_ground_plane(self, ray: Ray) -> RayHit:
        ground_plane = Plane(Vec3.from_xyz(0, 0, 1), Vec3.zero())
        return ground_plane.raycast(ray)

    def raycast_against_entity_obb(self, entity: RigidEntity, ray: Ray) -> RayHit:
        if isinstance(entity.morph, gs.morphs.Box):
            obb: OBB = self._get_box_obb(entity)
            ray_hit = obb.raycast(ray)
            if ray_hit.is_hit:
                ray_hit.geom = entity.geoms[0]
            return ray_hit
        elif isinstance(entity.morph, gs.morphs.Plane):
            # ignore plane
            return RayHit.no_hit()
        else:
            closest_hit = RayHit.no_hit()
            for link in entity.links:
                if not link.is_fixed: 
                    for geom in link.geoms:
                        obb: OBB = self._get_geom_placeholder_obb(geom)
                        ray_hit = obb.raycast(ray)
                        if ray_hit.distance < closest_hit.distance:
                            ray_hit.geom = geom
                            closest_hit = ray_hit
            return closest_hit

    def raycast_against_entities(self, ray: Ray) -> RayHit:
        closest_hit = RayHit.no_hit()
        for entity in self.scene.sim.rigid_solver.entities:
            rigid_entity: RigidEntity = cast(RigidEntity, entity)
            ray_hit = self.raycast_against_entity_obb(rigid_entity, ray)
            if ray_hit.distance < closest_hit.distance:
                closest_hit = ray_hit
        return closest_hit

    def _get_box_obb(self, box_entity: RigidEntity) -> OBB:
        box: gs.morphs.Box = box_entity.morph
        pose = Pose.from_link(box_entity.links[0])
        half_extents = 0.5 * Vec3.from_xyz(*box.size)
        return OBB(pose, half_extents)

    def _get_geom_placeholder_obb(self, geom: 'RigidGeom') -> OBB:
        pose = Pose.from_geom(geom)
        half_extents = Vec3.full(0.5 * 0.125)
        return OBB(pose, half_extents)

    def _draw_arrow(
        self, pos: Vec3, dir: Vec3, color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    ) -> None:
        self.scene.draw_debug_arrow(pos.v, dir.v, color=color)  # Only draws arrowhead -- bug?
        self.scene.draw_debug_line(pos.v, pos.v + dir.v, color=color)

    def _draw_entity_unrotated_obb(self, geom: 'RigidGeom') -> None:
        obb: OBB | None = None
        if isinstance(geom.entity.morph, gs.morphs.Box):
            obb = self._get_box_obb(geom.entity)
        else:
            obb = self._get_geom_placeholder_obb(geom)
        
        if obb:
            aabb: AABB = AABB.from_center_and_half_extents(obb.pose.pos, obb.half_extents)
            aabb.expand(padding=0.01)
            self.scene.draw_debug_box(aabb.v, color=Color.red().with_alpha(0.5).tuple(), wireframe=False)

        if True or isinstance(geom.entity.morph, gs.morphs.Box):
            self._draw_some_geometry_points(geom)

    # we know it's a box
    def _draw_some_geometry_points(self, geom: 'RigidGeom') -> None:
        return

        # get points
        verts = geom.get_verts() # in world space
        for i in range(verts.shape[0]):
            # pos = Vec3.from_tensor(verts[i])
            # self.scene.draw_debug_sphere(pos.v, 0.01, (1, 0, 0, 1))
            self.scene.draw_debug_sphere(verts[i], 0.01, (1, 0, 0, 1))

        # tris = geom.get_faces()
        # num_tris = min(tris.shape[0], 12)
        # for i in range(num_tris):
        #     self.scene.draw_debug_line(verts[tris[i, 0]], verts[tris[i, 1]], color=(0, 1, 0, 1))
        #     self.scene.draw_debug_line(verts[tris[i, 1]], verts[tris[i, 2]], color=(0, 1, 0, 1))
        #     self.scene.draw_debug_line(verts[tris[i, 2]], verts[tris[i, 0]], color=(0, 1, 0, 1))

        # geom.face_start same as geom._solver.geoms_info.face_start[geom.idx]

        face_end_tranctuated = min(geom.face_end, geom.face_start + 12)
        for face_idx in range(geom.face_start, face_end_tranctuated):
            assert geom._solver.faces_info.geom_idx[face_idx] == geom.idx
            verts_indices = gs.ti_ivec3(geom._solver.faces_info.verts_idx[face_idx])  # copy to modify
            verts_indices[0] -= geom.vert_start  # reverting offset added in _init_vert_fields
            verts_indices[1] -= geom.vert_start  # reverting offset added in _init_vert_fields
            verts_indices[2] -= geom.vert_start  # reverting offset added in _init_vert_fields

            self.scene.draw_debug_line(verts[verts_indices[0]], verts[verts_indices[1]], color=(0, 1, 0, 1))
            self.scene.draw_debug_line(verts[verts_indices[1]], verts[verts_indices[2]], color=(0, 1, 0, 1))
            self.scene.draw_debug_line(verts[verts_indices[2]], verts[verts_indices[0]], color=(0, 1, 0, 1))

        # Draw AABBs for all geometries
        if False:
            for entity in self.scene.sim.rigid_solver.entities:
                for geom in entity.geoms:
                    aabb: torch.Tensor = geom.get_AABB()
                    self.scene.draw_debug_box(aabb, color=Color.red().with_alpha(0.5).tuple(), wireframe=False)
                    ziu = 0


    def _raycast_geom(self, geom: 'RigidGeom', ray: Ray) -> RayHit:
        if not self.scene.is_built:
            return

        tri_idx = self._kernel_raycast_geom_in_local(geom, ti.math.vec3(*ray.origin.v), ti.math.vec3(*ray.direction.v))
        if -1 != tri_idx:
            self.scene.draw_debug_sphere(self.temp_rayhit_out[None].position, radius=0.05, color=(0, 0, 1, 1))

        
    @ti.kernel
    def _kernel_raycast_geom_in_local(self, geom: ti.template(), ray_origin: ti.math.vec3, ray_direction: ti.math.vec3) -> int:       
        b = 0  # batch index
        solver = geom._solver  # RigidSolver
        geoms_info = solver.geoms_info  # array_class.StructGeomsInfo
        geoms_state = solver.geoms_state  # array_class.StructGeomsState

        pos = geoms_state.pos[geom._idx, b]  # ti.math.vec3
        rot = geoms_state.quat[geom._idx, b]  # ti.math.vec4

        ray_origin_in_local = gu.ti_inv_transform_by_trans_quat(ray_origin, pos, rot)
        ray_direction_in_local = gu.ti_inv_transform_by_quat(ray_direction, rot)

        face_start = geoms_info.face_start[geom._idx]
        face_end = geoms_info.face_end[geom._idx]

        faces_info = solver.faces_info  # array_class.StructFacesInfo
        verts_info = solver.verts_info  # array_class.StructVertsInfo


        closest_t = 1e24
        closest_u = 0.0
        closest_v = 0.0
        closest_idx = -1        

        min_end = min(face_end, face_start + 1024)
        if min_end != face_end:
            print("Warning: Geom has more than 1024 faces. Only raycasting against the first 1024 tris.")
        for face_idx in range(face_start, face_end):
            assert faces_info.geom_idx[face_idx] == geom._idx
            verts_indices = gs.ti_ivec3(faces_info.verts_idx[face_idx])  # copy to modify

            v0 = verts_info.init_pos[verts_indices[0]]
            v1 = verts_info.init_pos[verts_indices[1]]
            v2 = verts_info.init_pos[verts_indices[2]]

            hit = self._func_raycast_triangle(v0, v1, v2, ray_origin_in_local, ray_direction_in_local)
            t = hit.x
            if t > 0.0 and t < closest_t:
                closest_t = t
                closest_u = hit.y
                closest_v = hit.z
                closest_idx = face_idx            

        if -1 != closest_idx:
            point_in_world = ray_origin + closest_t * ray_direction
            normal = ti.math.vec3(0.0, 0.0, 0.0)

            ray_hit = tiRayHit(
                position=point_in_world,
                normal=normal,
                distance=closest_t
            )
            self.temp_rayhit_out[None] = ray_hit

        return closest_idx

        # verts = torch.empty((vend - vstart, 3), dtype=gs.tc_float, device=gs.device)
        # verts = geom.func_get_verts(t)
        # return self.raycast_against_verts(verts, ray)

    # ChatGPT: using Möller–Trumbore algorithm
    # It returns a ti.math.vec3 where:
    #   result.x = t (intersection distance)
    #   result.y = u (barycentric coordinate)
    #   result.z = v (barycentric coordinate)
    # If no hit, you can return a sentinel value (e.g. vec3(-1.0, 0.0, 0.0)).
    @ti.func
    def _func_raycast_triangle(self, a: ti.math.vec3, b: ti.math.vec3, c: ti.math.vec3,
                            ray_origin: ti.math.vec3, ray_direction: ti.math.vec3) -> ti.math.vec3:
        result = ti.math.vec3(-1.0, 0.0, 0.0)  # default = no hit
        epsilon = 1e-6

        edge1 = b - a
        edge2 = c - a
        h = ray_direction.cross(edge2)
        det = edge1.dot(h)

        if ti.abs(det) > epsilon:
            inv_det = 1.0 / det
            s = ray_origin - a
            u = s.dot(h) * inv_det

            if 0.0 <= u <= 1.0:
                q = s.cross(edge1)
                v = ray_direction.dot(q) * inv_det

                if v >= 0.0 and (u + v) <= 1.0:
                    t = edge2.dot(q) * inv_det
                    if t > epsilon:
                        result = ti.math.vec3(t, u, v)

        return result        