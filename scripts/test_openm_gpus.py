#!/usr/bin/env python3
import os
import traceback
import openmm as mm
from openmm import app, unit

def make_minimal_system():
    system = mm.System()
    system.addParticle(39.9 * unit.amu)

    force = mm.CustomExternalForce("0.5*k*(x*x+y*y+z*z)")
    force.addGlobalParameter("k", 1.0 * unit.kilojoule_per_mole / unit.nanometer**2)
    force.addParticle(0, [])
    system.addForce(force)

    top = app.Topology()
    chain = top.addChain()
    res = top.addResidue("ARG", chain)
    top.addAtom("CA", app.Element.getByAtomicNumber(6), res)

    return system, top

def test_gpu(device_index):
    system, top = make_minimal_system()
    integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
    platform = mm.Platform.getPlatformByName("CUDA")

    props = {
        "DeviceIndex": str(device_index),
        "Precision": "mixed",
    }

    sim = app.Simulation(top, system, integrator, platform, props)
    sim.context.setPositions([[0.0, 0.0, 0.0]] * unit.nanometer)

    state = sim.context.getState(getEnergy=True)
    energy = state.getPotentialEnergy()

    actual_device = sim.context.getPlatform().getPropertyValue(
        sim.context, "DeviceIndex"
    )

    print(f"[OK] requested GPU {device_index}, actual DeviceIndex={actual_device}, E={energy}")

def main():
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("OpenMM version =", mm.version.version)
    print()

    n = int(os.environ.get("N_GPUS_TO_TEST", "4"))

    for i in range(n):
        print(f"=== Testing visible GPU index {i} ===")
        try:
            test_gpu(i)
        except Exception:
            print(f"[FAIL] GPU {i}")
            traceback.print_exc()
        print()

if __name__ == "__main__":
    main()
