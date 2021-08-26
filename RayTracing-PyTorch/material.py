import random
from ray import vec3, ray, dot
import math
import numpy as torch
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
        r_out_parallel = n * (-torch.sqrt(torch.abs(1.0 - r_out_perp.dot(r_out_perp))))
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
                             is_scattered = torch.full(len(r_in), True, dtype=torch.bool))

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
                             is_scattered = torch.full(len(r_in), True, dtype=torch.bool))

class dielectric(material):
    def __init__(self, ref_idx):
        self.ref_idx = ref_idx

    def scatter(self, r_in, rec):

        ni_over_nt = torch.where(rec.front_face, 1.0/self.ref_idx, self.ref_idx)

        unit_direction = r_in.direction.unit_vec()

        cos_theta = torch.fmin(dot(unit_direction*-1.0, rec.normal), 1.0)
        sin_theta = torch.sqrt(1.0 - cos_theta*cos_theta)
        reflected = self.reflect(unit_direction, rec.normal)
        refracted = self.refract(unit_direction, rec.normal, ni_over_nt)

        reflected_rays = ray(rec.p, reflected)
        refracted_rays = ray(rec.p, refracted)

        reflect_prob = self.schlick(cos_theta, ni_over_nt)
        random_floats = my_random(0.0, 1.0, len(reflect_prob))

        must_reflect = (ni_over_nt * sin_theta > 1.0)
        again_reflect = (random_floats < reflect_prob)

        all_reflect = torch.where(torch.logical_or(must_reflect, again_reflect))
        
        refracted_rays[all_reflect] = reflected_rays[all_reflect]
        
        return ScatterResult(attenuation = vec3(1.0, 1.0, 1.0),
                             rays = refracted_rays,
                             is_scattered = torch.full(len(r_in), True, dtype=torch.bool))

        """

        cos = dot(unit_direction, rec.normal)
        disc = (1.0 - (1-cos*cos)*(ni_over_nt*ni_over_nt))
        refracted = self.refract(unit_direction, outward_normal, ni_over_nt, cos, disc)

        cos = torch.where(rec.front_face, cos*self.ref_idx, cos*-1)

        attenuation = vec3(1.0, 1.0, 1.0)
        
        reflected = self.reflect(unit_direction, rec.normal)
        reflect_prob = self.schlick(cos)
        r = torch.random.rand(len(r_in))

        must_reflect = disc < 0
        again_reflect = r < reflect_prob

        all_reflect = torch.where(torch.logical_or(must_reflect, again_reflect))

        refracted[all_reflect] = reflected[all_reflect]


        return ray(rec.p, refracted), attenuation, torch.full(len(r_in), True, dtype=torch.bool)

        """
    def schlick(self, cos, ref_idx):
        r0 = (1.0 - ref_idx) / (1.0 + self.ref_idx)
        r0 = r0*r0
        return r0 + (1 - r0) * (1-cos)**5
    


def random_in_unit_sphere(n):
    u = torch.random.rand(n).astype(torch.float32)
    v = torch.random.rand(n).astype(torch.float32)
    w = torch.random.rand(n).astype(torch.float32)

    cr = torch.cbrt(w)

    z = cr*(-2.0*u+1.0)
    sq_z = torch.sqrt(1.0-z**2.0)
    x = cr*sq_z*torch.cos(2.0*torch.pi*v)
    y = cr*sq_z*torch.sin(2.0*torch.pi*v)

    return vec3(x,y,z)

def random_unit_vectors(n):
    a = my_random(0.0, 2.0*torch.pi, n)
    z = my_random(-1.0, 1.0, n)
    r = torch.sqrt(1.0 - z*z)
    return vec3(r*torch.cos(a), r*torch.sin(a), z)


def my_random(low, high, size):
    return torch.random.uniform(low, high, size).astype(torch.float32)