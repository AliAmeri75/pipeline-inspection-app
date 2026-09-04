def CorLAS(SF, sigmaY, sigmaU, E, CVN, flow_definition, D, WT, flaw_length, flaw_depth, shape, flaw_location, Fd, Ft):
    import numpy as np
    pi=np.pi
# Calculates the failure pressure based on the flow stress criterion
# Calculate flow strength
    if (flow_definition =='1e4'):
      sigmaFL=sigmaY+10000;
    elif (flow_definition=='Other'):
      sigmaFL=(sigmaY+sigmaU)/2;

    # Calculate y factor
    if D/(2*WT)<=10:
       y=0.4;
    elif D/(2*WT)> 10:
       y=0;

    # Calculate reference area
    A0=flaw_length*WT; 
    A_Eff=flaw_length*flaw_depth*(int(shape=='R'))+pi*flaw_length*flaw_depth/4*(int(shape=='E'));
    A=flaw_depth*(int(shape=='R'))+4*A_Eff/(pi*flaw_length)*(int(shape=='E'));
    # Calculate Folias factor
    M=((1+0.6275*flaw_length**2/(D*WT)-0.003375*flaw_length**4/(D*WT)**2)**0.5)*int(flaw_length** 2/(D*WT)<= 50)+(3.3+0.032*flaw_length**2/(D*WT))*int(flaw_length**2/(D*WT)>50);
    # Calculate failure pressure
    Pressure_f1=Fd*Ft*sigmaFL*((1-A_Eff/A0)/(1-A_Eff/(M*A0)))/(D/(2*WT)-y);
    ## Calculates the failure pressure based on the fracture toughness criterion
    # Calculate strain hardening exponent
    n=-0.00546+0.556*(sigmaY/sigmaU)-0.547*(sigmaY/sigmaU)**2; 
    # Calculate F3 Factor
    F3=(3.85*(1/n)**0.5*(1-n)+pi*n)*(1+n); 
    # Calculate K Coefficient
    k=(0.005-sigmaY/(E))/sigmaY**(1/n); 
    # Calculate critical value of J
    Jc=round(12*CVN/0.124);
    # Calculate effective flaw area depending on rectangular or elliptical
    if A/flaw_length>0.5:
       z=0.5;
    else:
       z=A/flaw_length;

    # Calculate shape factor 
    Qf=1.2581-0.20589*(z)-11.493*(z)**2+29.586*(z)**3-23.584*(z)**4; 
    # Calculate free surface factor
    if A/WT<=0.95:
       Fsf=(2*WT/(pi*A))*np.tan(pi*A/(2*WT))*(1-2*z)+2*z;
    elif A/WT>0.95:
       Fsf=(8.515+(A/WT-0.95)*(162/WT))*(1-2*z)+2*z;

    # Calculate folias factor
    if flaw_length**2/(D*WT)<=50:
       M=(1+0.6275*flaw_length**2/(D*WT)-0.003375*flaw_length**4/(D*WT)**2)**0.5;
    elif flaw_length**2/(D*WT)>50:
       M=3.3+0.032*flaw_length**2/(D*WT);

    # Iteration to calculate signma_normal via Newton Raphson method
    maxstep=1000;
    step=1;
    sigma=sigmaY;
    delta_sigma=1000;
    Ji=Qf*Fsf*A*(sigma**2*pi/E+F3*k*sigma**(1/n)*sigma)-Jc;
    dJi=Qf*Fsf*A*(2*pi*sigma/E+(1+1/n)*F3*k*sigma**(1/n));
    while abs(Ji)>1e-3 and step<=maxstep:
       dJi=Qf*Fsf*A*(2*pi*sigma/E+(1+1/n)*F3*k*sigma**(1/n));
       delta_sigma=-Ji/dJi;
       sigma=sigma+delta_sigma;
       Ji=Qf*Fsf*A*(sigma**2*pi/E+F3*k*sigma**(1/n)*sigma)-Jc;
       step=step+1;
    # Compute normal stress and failure pressure
    sigmaN=sigma*(1-A_Eff/A0)/(1-A_Eff/(A0*M));
    if (flaw_location=='I')or(flaw_location=='ND'):
       Pressure_f2=sigmaN*WT/((pi/4)*A+0.5*D);
    elif (flaw_location=='E'):
       Pressure_f2=2*sigmaN*WT/D;

    # The lower one will be failure pressure
    Pressure_fd=min(Pressure_f1,Pressure_f2);
    # Calculate safety factor
    MOP=Pressure_fd/SF;
    return [Pressure_fd, MOP,Qf,Fsf,M]
