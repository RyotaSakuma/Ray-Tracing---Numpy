from ray import vec3, ray, dot
import math
import torch
from material import metal, lambertian, dielectric
import random

class hit_record():
    def __init__(self, n):
        self.t = torch.full(n, float('inf'), dtype=torch.float32)
        self.p = vec3(torch.empty(n, dtype=torch.float32), torch.empty(n, dtype=torch.float32), torch.empty(n, dtype=torch.float32))
        self.normal = vec3(torch.empty(n, dtype=torch.float32), torch.empty(n, dtype=torch.float32), torch.empty(n, dtype=torch.float32))
        self.front_face = torch.zeros(n, dtype=torch.float32)
        self.index = torch.arange(n, dtype=torch.int32)
        self.mat_id = torch.zeros(n, dtype=torch.int64)

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

    def hit(self, r, t_min, t_max, rec):
        oc = r.origin - self.center
        a =r.direction.dot(r.direction)
        b = r.direction.dot(oc)
        c = oc.dot(oc) - self.radius*self.radius

        disc = b**2 - a*c

        sq = torch.sqrt(torch.maximum(0., disc))
        t1 = (-b-sq) / a
        t2 = (-b+sq) / a

        hit1 = torch.logical_and(t1 < t_max, t1 > t_min)
        hit2 = torch.logical_and(t2 < t_max, t2 > t_min)
        hit = disc >= 0.

        t = torch.where(hit2, t2, float('inf'))
        t = torch.where(hit1, t1, t)
        t = torch.where(hit, t, float('inf'))

        m = torch.where(t < rec.t)
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