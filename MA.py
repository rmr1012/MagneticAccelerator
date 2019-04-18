import math
import numpy as np
u0=4*math.pi*10**-7 #vacuum permeability
rpoCopper=1.68*10**-8
class Coil:
    #default values
    L=0  # uH
    R=0  # Ohm
    n=0  # turns
    width=0   #mm
    innerDia=5  #mm
    outerDia=5  #mm
    gauge=20  #AWG
    AWGLUT={1:7.3481,2:6.5437,3:5.8273,4:5.1894,5:4.6213,6:4.1154,7:3.6649,8:3.2636,9:2.9064,10:2.5882,11:2.3048,12:2.0525,13:1.8278,14:1.6277,15:1.4495,16:1.2908,17:1.1495,18:1.0237,19:0.9116,20:0.8118,21:0.7229,22:0.6438,23:0.5733,24:0.5106,25:0.4547,26:0.4049,27:0.3606,28:0.3211,29:0.2859,30:0.2546,31:0.2268,32:0.2019,33:0.1798,34:0.1601,35:0.1426,36:0.1270,37:0.1131,38:0.1007,39:0.0897,40:0.0799}
    def __init__(self,n,innerDia,width,gauge): #input, n:turns, innerDia:inner diameter(mm), length(mm), gauge(awg)
        self.n=n
        self.width=width
        self.innerDia=innerDia
        self.gauge=gauge
        self.wireDia=self.AWGLUT[self.gauge]
        self.tpl=math.ceil(self.width/self.wireDia)
        self.layers=math.ceil(self.n/self.tpl)
        self.outerDia=self.innerDia+2*(self.layers*self.wireDia)

        remainingTurns=self.n
        self.wireLength=0
        for layer in range(self.layers):
            D=self.innerDia+(layer+1)*self.wireDia
            if remainingTurns>= self.tpl:
                LayerL=self.tpl**2*u0*(D*10**-3/2)*(np.log(8*D/self.wireDia)-2)
                layerl=D*math.pi*self.tpl
            else:
                LayerL=remainingTurns**2*u0*(D*10**-3/2)*(np.log(8*D/self.wireDia)-2)
                layerl=D*math.pi*remainingTurns
            print(LayerL, layerl)
            self.L+= LayerL
            self.wireLength +=layerl
            remainingTurns=remainingTurns-self.tpl
        self.R=rpoCopper*self.wireLength*10**-3/(math.pi*(self.wireDia*10**-3/2)**2)
        # self.L=(d^2)/()
        # self.R=0
