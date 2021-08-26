import random
from ray import vec3, ray, dot
import math
import numpy as np
from collections import namedtuple

ScatterResult = namedtuple('ScatterResult', 'attenuation rays is_scattered')

class material():
    def scatter(self, r_in, rec):
        return

    def reflect(self, v, n):
        return v - n * (dot(v, n)) * 2.0

    def refract(self, uv, n, ni_over_nt):
        cos_theta = dot(uv*-1.0, n)
        r_out_perp = (uv + n*cos_theta) * ni_over_nt
        r_out_parallel = n * (-np.sqrt(np.abs(1.0 - r_out_perp.dot(r_out_perp))))
        return r_out_perp + r_out_parallel


class lambertian(material):
    def __init__(self, albedo):
        super().__init__()
        self.albedo = albedo

    def scatter(self, r_in, rec):
        target = rec.normal + random_in_unit_sphere(len(r_in))
        scattered = ray(rec.p, target)
        return ScatterResult(attenuation = self.albedo,
                             rays = scattered,
                             is_scattered = np.full(len(r_in), True, dtype=np.bool))

class metal(material):
    def __init__(self, albedo, fuzz=0.0):
        super().__init__()
        self.albedo = albedo
        self.fuzz = fuzz

    def scatter(self, r_in, rec):
        target = rec.normal + random_in_unit_sphere(len(r_in))

        reflected = self.reflect(r_in.direction.unit_vec(), rec.normal)
        scattered = ray(rec.p, reflected + random_in_unit_sphere(len(r_in)) * self.fuzz)

        return ScatterResult(attenuation = self.albedo,
                             rays = scattered,
                             is_scattered = dot(scattered.direction, rec.normal) > 0)

class dielectric(material):
    def __init__(self, ref_idx):
        self.ref_idx = ref_idx

    def scatter(self, r_in, rec):

        ni_over_nt = np.where(rec.front_face, 1.0/self.ref_idx, self.ref_idx)

        unit_direction = r_in.direction.unit_vec()

        cos_theta = np.fmin(dot(unit_direction*-1.0, rec.normal), 1.0)
        sin_theta = np.sqrt(1.0 - cos_theta*cos_theta)
        reflected = self.reflect(unit_direction, rec.normal)
        refracted = self.refract(unit_direction, rec.normal, ni_over_nt)

        reflected_rays = ray(rec.p, reflected)
        refracted_rays = ray(rec.p, refracted)

        reflect_prob = self.schlick(cos_theta, ni_over_nt)
        random_floats = my_random(0.0, 1.0, len(reflect_prob))

        must_reflect = (ni_over_nt * sin_theta > 1.0)
        again_reflect = (random_floats < reflect_prob)

        all_reflect = np.where(np.logical_or(must_reflect, again_reflect))
        
        refracted_rays[all_reflect] = reflected_rays[all_reflect]
        
        return ScatterResult(attenuation = vec3(1.0, 1.0, 1.0),
                             rays = refracted_rays,
                             is_scattered = np.full(len(r_in), True, dtype=np.bool))

    def schlick(self, cos, ref_idx):
        r0 = (1.0 - ref_idx) / (1.0 + self.ref_idx)
        r0 = r0*r0
        return r0 + (1 - r0) * (1-cos)**5
    


def random_in_unit_sphere(n):
    u = np.random.rand(n).astype(np.float32)
    v = np.random.rand(n).astype(np.float32)
    w = np.random.rand(n).astype(np.float32)

    cr = np.cbrt(w)

    z = cr*(-2.0*u+1.0)
    sq_z = np.sqrt(1.0-z**2.0)
    x = cr*sq_z*np.cos(2.0*np.pi*v)
    y = cr*sq_z*np.sin(2.0*np.pi*v)

    return vec3(x,y,z)