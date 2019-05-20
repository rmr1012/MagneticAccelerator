from descrete_optimization import *


c=Capacitor(15,0.00065,0)
p=Projectile(0.015,0.005,saturation=0.75)
s=Solenoid(2000,0.02,inner_dia=0.005,gauge=28)
print(s.inductance)
laStage=Stage(s,c,p,offset=0.01)
effi=laStage.calculate_efficiency(0.001,1000)


# myCoil=Coil(2000,5,20,28)#n,innerDia,width,gauge
# print(myCoil.L)
# ironBall=Bullet(15,5,5) #m,length,diameter,ur=6.3*10**-3,saturation=0.75):
# stage1Cap=Cap(15,650) #,voltage,capacitance,esr=0):
