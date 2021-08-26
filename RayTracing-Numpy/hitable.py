from ray import vec3, ray, dot
import math
import numpy as np
from material import metal, lambertian, dielectric
import random

class hit_record():
    def __init__(self, n):
        self.t = np.full(n, np.inf, dtype=np.float32)
        self.p = vec3(np.empty(n, dtype=np.float32), np.empty(n, dtype=np.float32), np.empty(n, dtype=np.float32))
        self.normal = vec3(np.empty(n, dtype=np.float32), np.empty(n, dtype=np.float32), np.empty(n, dtype=np.float32))
        self.front_face = np.zeros(n, dtype=np.float32)
        self.index = np.arange(n, dtype=np.int32)
        self.mat_id = np.zeros(n, dtype=np.int64)

    def __getitem__(self, idx):
        other = hit_record(n = len(idx))
        other.t = self.t[idx]
        other.p = self.p[idx]
        other.normal = self.normal[idx]
        other.front_face = self.front_face[idx]
        other.index = self.index[idx]
        other.mat_id = self.mat_id[idx]

        return other

    def __setitem__(self, idx, other):
        self.t[idx] = other.t
        self.p[idx] = other.p
        self.normal[idx] = other.normal
        self.front_face[idx] = other.front_face
        self.index = other.index
        self.mat_id = other.mat_id

        
class hitable():
    def hit(self, r, t_min, t_max):
        return

class sphere(hitable):
    def __init__(self, center, radius, material):
        self.center = center
        self.radius = radius
        self.mat = material
        self.bounding_box = AABB(center - vec3(radius, radius, radius), 
                                center + vec3(radius, radius, radius))

    def hit(self, r, t_min, t_max, rec):
        oc = r.origin - self.center
        a =r.direction.dot(r.direction)
        b = r.direction.dot(oc)
        c = oc.dot(oc) - self.radius*self.radius

        disc = b**2 - a*c

        sq = np.sqrt(np.maximum(0., disc))
        t1 = (-b-sq) / a
        t2 = (-b+sq) / a

        hit1 = np.logical_and(t1 < t_max, t1 > t_min)
        hit2 = np.logical_and(t2 < t_max, t2 > t_min)
        hit = disc >= 0.

        t = np.where(hit2, t2, np.inf)
        t = np.where(hit1, t1, t)
        t = np.where(hit, t, np.inf)

        m = np.where(t < rec.t)
        if len(m[0])==0:
            return

        rays = r[m]

        p = rays.point_at_parameter(t[m])
        outward_normal = (p - self.center) / self.radius
        front_face = dot(rays.direction, outward_normal) < 0
        normal = vec3.where(front_face, outward_normal, outward_normal*-1.0)

        rec.t[m]= t[m]
        rec.p[m] = p
        rec.normal[m] = normal
        rec.mat_id[m] = id(self.mat)
        rec.front_face[m] = front_face

class AABB():
    def __init__(self, min, max):
        self.v_min = min
        self.v_max = max

    def hit(self, rays, t_min, t_max):
        t0_x = np.fmin((self.v_min.x - rays.origin.x) / rays.direction.x,
                        (self.v_max.x - rays.origin.x) / rays.direction.x)
        t0_y = np.fmin((self.v_min.y - rays.origin.y) / rays.direction.y,
                        (self.v_max.y - rays.origin.y) / rays.direction.y)
        t0_z = np.fmin((self.v_min.z - rays.origin.z) / rays.direction.z,
                        (self.v_max.z - rays.origin.z) / rays.direction.z)
        t1_x = np.fmax((self.v_min.x - rays.origin.x) / rays.direction.x,
                        (self.v_max.x - rays.origin.x) / rays.direction.x)
        t1_y = np.fmax((self.v_min.y - rays.origin.y) / rays.direction.y,
                        (self.v_max.y - rays.origin.y) / rays.direction.y)
        t1_z = np.fmax((self.v_min.z - rays.origin.z) / rays.direction.z,
                        (self.v_max.z - rays.origin.z) / rays.direction.z)

        t0_x = np.fmax(t0_x, t_min)
        t0_y = np.fmax(t0_y, t_min)
        t0_z = np.fmax(t0_z, t_min)
        t1_x = np.fmin(t1_x, t_max)
        t1_y = np.fmin(t1_y, t_max)
        t1_z = np.fmin(t1_z, t_max)

        return np.logical_and(t0_x < t1_x, t0_y < t1_y, t0_z < t1_z)        