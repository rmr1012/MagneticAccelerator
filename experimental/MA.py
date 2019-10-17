import math
import numpy as np
import matplotlib.pyplot as plt
u0=4*math.pi*10**-7 #vacuum permeability
rpoCopper=1.68*10**-8

VERBOSE= False
class Cap:
    v0=0
    v=0 # time domain vector
    c=0 #uF
    esr=0
    def __init__(self,voltage,capacitance,esr=0):
        self.esr=esr
        self.v0=voltage
        self.v=self.v0
        self.c=capacitance
    def getEnergy(self):
        return 0.5*(self.c*10**-6)*self.v**2
class Bullet:
    d=0
    v=0
    a=0
    def __init__(self,m,length,diameter,ur=6.3*10**-3,saturation=0.75):
        self.m=m #grams
        self.length=length #mm
        self.diameter=diameter #mm
        self.ur=ur #relative permeability
        self.caliber=np.pi*(self.diameter/2)**2  # crosssection area
        self.saturation=saturation   #saturation (Tesla)
    def getMaxForce(self):
        return self.saturation**2*self.caliber*10**-6/2/u0

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
        self.tpl=math.ceil(self.width/self.wireDia) # turns per layer
        self.width=self.tpl*self.wireDia # actual width
        self.layers=math.ceil(self.n/self.tpl)
        self.outerDia=self.innerDia+2*(self.layers*self.wireDia)    # return inner_dia + 2 * layer * wire_dia

        remainingTurns=self.n
        self.wireLength=0
        for layer in range(self.layers):
            D=self.innerDia+2*layer*self.wireDia
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
        self.L=self.L * 10**6 # H to uH
        self.R=rpoCopper*self.wireLength*10**-3/(math.pi*(self.wireDia*10**-3/2)**2)




class Stage():
    def __init__(self,coil,capacitor,bullet,offset):
        self.coil=coil
        self.capacitor=capacitor
        self.offset=offset
        self.bullet=bullet

    def simulate(self,duration,runcycle,plot=False):
        timestep=duration/runcycle;

        tTrain=np.linspace(0,duration,runcycle)
        Icyc=np.zeros(runcycle)
        Idis=np.zeros(runcycle)
        Iflywheel=np.zeros(runcycle)
        force=np.zeros(runcycle)
        force_raw=np.zeros(runcycle)
        dis=np.zeros(runcycle)
        v=np.zeros(runcycle)
        acc=np.zeros(runcycle)

        a=(self.capacitor.esr+self.coil.R)/(2*self.coil.L*10**-6)
        w=1/np.sqrt(self.coil.L*10**-6*self.capacitor.c*10**-6)

        maxforce=self.bullet.getMaxForce()
        B1=0
        B2=self.capacitor.v0/(self.coil.L*10**-6)/w
        C1=self.capacitor.v0
        C2=0
        Icyc = B1*np.exp(-a*tTrain)*np.cos(w*tTrain)+B2*np.exp(-a*tTrain)*np.sin(w*tTrain)
        Idis = max(Icyc)*np.exp((-self.coil.R*tTrain)/(self.coil.L*10**-6))

        # generate diode flywheel response
        insersionPt=0
        captured=0
        for i in range(runcycle-1):
            if (Icyc[i]>Icyc[i+1] and captured ==0):
                insersionPt=i
                captured=1
        Iflywheel=Icyc
        Iflywheel[insersionPt:runcycle]=Idis[insersionPt:]

        #simulate force transient
        iterr = iter(range(runcycle))
        next(iterr) # skip first term
        for i in iterr:
            if dis[i-1]*1000<self.offset:
                force[i]=(self.coil.n*Iflywheel[i])**2*u0*self.bullet.caliber*10**-6/2/((self.offset*10**-3-dis[i-1]))**2
                if VERBOSE:
                    print('before %.4f %.4f %d',dis[i-1],force[i],i)
            elif (self.offset-dis[i-1]*1000 <= 0 and dis[i-1]*1000-self.offset <= self.coil.width ): # if in coil
                force[i]=(.5-((dis[i-1]*10**3-self.offset)/self.coil.width))*2* (self.bullet.ur*u0*self.coil.n*Iflywheel[i])**2*self.bullet.caliber*10**-6/2/u0
                if VERBOSE:
                    print('within %.4f %.4f %d',dis[i-1],.5-((dis[i-1]*10**3-self.offset)/self.coil.width),i)
            elif dis[i-1]*1000-self.offset-self.coil.width>0:
                force[i]=-(self.coil.n*Iflywheel[i])**2*u0*self.bullet.caliber*10**-6/2/((dis[i-1]-self.offset*10**-3-self.coil.width*10**-3))**2
                if VERBOSE:
                    print('beyond %.4f %.4f %d',dis[i-1],force[i],i)

            force_raw[i]=force[i];
            if force[i]>0:
                force[i] = min(force[i],maxforce);
            else:
                force[i] = -min(-force[i],maxforce);

            acc[i]=force[i]/(self.bullet.m*10**-3);
            v[i]=v[i-1]+acc[i]*timestep;
            dis[i]=dis[i-1]+v[i]*timestep;
        plt.plot(Iflywheel)
        plt.show()
    #     iterr = iter(range(runcycle))
    #     next(iterr) # skip first term
    #     for i in iterr:
    #         force_raw[i]=self.getRawForce(dis[i-1]-self.offset,Iflywheel[i])
    #         if force_raw[i]>0:
    #             force[i] = min(force_raw[i],maxforce);
    #         else:
    #             force[i] = -min(-force_raw[i],maxforce);
    #
    #         a[i]=force[i]/(self.bullet.m*10**-3);
    #         v[i]=v[i-1]+a[i]*timestep;
    #         dis[i]=dis[i-1]+v[i]*timestep;
    #
        outenergy=0.5*(self.bullet.m*10**-3)*v[-1]**2;
        inenergy=self.capacitor.getEnergy()
        return outenergy/inenergy
    #
    # def getRawForce(self,dis,I): # dis datums at coil start
    #     if dis*1000<0:
    #         force=(self.coil.n*I)**2*u0*self.bullet.caliber*10**-6/2/((-dis))**2
    #     elif (dis*1000 <= self.coil.width): # if in coil
    #         force=(.5-((dis*10**3)/coil.width))*2* (self.bullet.ur*u0*self.coil.n*I)**2*self.bullet.caliber*10**-6/2/u0
    #     elif (dis*1000 >  self.coil.width):
    #         force=-(self.coil.n*I)**2*u0*self.bullet.caliber*10**-6/2/((dis-self.coil.width*10**-3))**2
    #     return force
if __name__=="__main__":
    #
    myCoil=Coil(100,5,25,22)#n,innerDia,width,gauge
    print("L: "+ str(myCoil.L)+"uH")
    print("R: "+ str(myCoil.R)+"Ohms")
#     print(myCoil.L)
#     ironBall=Bullet(2.9,19,4.76) #m,length,diameter,ur=6.3*10**-3,saturation=0.75):
#     #2.9g, 19mm long, 4.76 mm diameter
#     stage1Cap=Cap(20,1200,0.1) #,voltage,capacitance,esr=0):

#     stage1=Stage(myCoil,stage1Cap,ironBall,1) #1mm offset
#     effi=stage1.simulate(0.01,1000)
#     print(effi)

#     xsweep=np.linspace(20,250,100)
#     ysweep=np.linspace(-1,5,100)
#     resultMatrix=np.zeros([100,100])
#     for xindex,n in enumerate(xsweep):
#         for yindex,offset in enumerate(ysweep):
#             resultMatrix[xindex,yindex][]
