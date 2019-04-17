%% Start by simulating the behavior of 1 stage with given parameters
% effi = function(gauge,n,C,D,V0) 
runcycle=1000
runduration=0.10
L= 50000%in uH
C= 650%in uF
n= 1000 % winding count
Rcoil= 15 % in ohms
Rpara= 1 
R=Rcoil+Rpara

a=R/(2*L*10^-6)
w=1/sqrt(L*10^-6*C*10^-6)

V0=15 % in Volts
I0=0

B1=I0
B2=V0/(L*10^-6)/w

C1=V0
C2=0

coil_length=10 %mm
coil_radius=5 %mm
caliber = 25 %mm^2(5mm by 5mm)
load_length=3 %mm
load_mass=15 % grams
u0=4*pi*10^-7 %vacuum permeability
u=6.3*10^-3 %iron permeability (H/m)
saturation=0.75 %Teslas
D=1 %mm starting offset
% Vc=zeros(runcycle,1)
% Vc(1:2)=Vinit
% %Model 1 regular RLC
% s=tf('s')
% vcc=(-R/(s*L))/(1-(1/(s^2*L*C)))
Icyc=zeros(runcycle,1)
Idis=zeros(runcycle,1)
Iflywheel=zeros(runcycle,1)

Vcyc=zeros(runcycle,1)

t=linspace(0,runduration,runcycle)

Icyc = B1.*exp(-a.*t).*cos(w.*t)+B2.*exp(-a.*t).*sin(w.*t)
Idis = max(Icyc)*exp((-R.*t)/(L*10^-6))
%Vcyc = C1.*exp(-a.*t).*sin(w.*t)+C2.*exp(-a.*t).*cos(w.*t)
insersion=0
captured=0
for i=1:runcycle-1
    if (Icyc(i)>Icyc(i+1) && captured ==0)
        insersion=i
        captured=1
    end
end
Iflywheel=Icyc
Iflywheel(insersion:runcycle)=Idis(1:runcycle-insersion+1)
%Iflywheel=ones(runcycle,1) %eq cal trick

% plot(t,Icyc)
% hold on 
% plot(t,Iflywheel)
% Calculate b field
test_force=(n*Iflywheel).^2*u0*caliber*10.^-6/2/(D*10^-3)^2;
maxforce = saturation^2*caliber*10.^-6/2/u0
%Btot=(u0*Icyc*n/2)*(((D*10^-3+coil_length*10^-3)/sqrt((D*10^-3+coil_length*10^-3)^2+(coil_radius*10^-3)^2))-((D*10^-3)/sqrt((D*10^-3)^2+(coil_radius*10^-3)^2)))
force=zeros(runcycle,1);
force_raw=zeros(runcycle,1);
dis=zeros(runcycle,1);
v=zeros(runcycle,1);
a=zeros(runcycle,1);

timestep=runduration/runcycle;
for i=2:runcycle
    if dis(i-1)*1000<D
        force(i)=(n*Iflywheel(i)).^2*u0*caliber*10.^-6/2/((D*10^-3-dis(i-1)))^2;
        disp(sprintf('before %.4f %.4f %d',dis(i-1),force(i),i))
    elseif (D-dis(i-1)*1000 <= 0 && dis(i-1)*1000-D <= coil_length ) % if in coil
        force(i)=(.5-((dis(i-1)*10^3-D)/coil_length))*2* (u*n*Iflywheel(i))^2*caliber*10.^-6/2/u0;
        disp(sprintf('within %.4f %.4f %d',dis(i-1),.5-((dis(i-1)*10^3-D)/coil_length),i))
    elseif dis(i-1)*1000-D-coil_length>0
        force(i)=-(n*Iflywheel(i)).^2*u0*caliber*10.^-6/2/((dis(i-1)-D*10^-3-coil_length*10^-3))^2;
        disp(sprintf('beyond %.4f %.4f %d',dis(i-1),force(i),i))
    end
    force_raw(i)=force(i);
    if force(i)>0
        force(i) = min(force(i),maxforce);
    else
        force(i) = -min(-force(i),maxforce);
    end
    a(i)=force(i)/(load_mass*10^-3);
    v(i)=v(i-1)+a(i)*timestep;
    dis(i)=dis(i-1)+v(i)*timestep;
end
plot(v)
outenergy=0.5*(load_mass*10^-3)*v(end)^2;
inenergy=0.5*(C*10^-6)*V0^2;
effi=100*outenergy/inenergy
disp(v(end))