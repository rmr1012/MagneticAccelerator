from descrete_optimization import Solenoid
from descrete_optimization import Capacitor
from descrete_optimization import Projectile


class Stage:
    def __init__(self, solenoid: Solenoid, capacitor: Capacitor,
                 projectile: Projectile):
        self._solenoid = solenoid
        self._capacitor = capacitor
        self._projectile = projectile

    def simulate(self, duration, runcycle):
        pass

