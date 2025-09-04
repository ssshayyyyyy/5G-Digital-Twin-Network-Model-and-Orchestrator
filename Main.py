import matlab.engine
from pathlib import Path

PROJECT_DIR = Path(r"C:\Users\ASUS\Desktop\硕士毕设\matlab代码\Topo_5Gsimulink_handover.slx").parent
MODEL       = 'Topo_5Gsimulink_handover'

def main():
    eng = matlab.engine.start_matlab()
    eng.cd(str(PROJECT_DIR), nargout=0)
    eng.load_system(MODEL, nargout=0)

    # parameters injection
    eng.workspace['amp']          = 30
    eng.workspace['amf_state']    = 1
    eng.workspace['gnb1_status']  = 1
    eng.workspace['gnb2_status']  = 1

    #variables check to make sure everyone is here
    simout = eng.sim(MODEL, 'StopTime', '10',nargout=1)
    print('MATLAB variables：', eng.eval('who', nargout=1))

    #read the result, check if it's same as the MATLAB results
    cs1 = eng.get(simout, 'cs1')
    best = eng.get(simout, 'best')
    print('cs1  10:', cs1[:10])
    print('best 10:', best[:10])

    import matplotlib.pyplot as plt
    import numpy as np

    cs1_check = np.array(cs1).flatten()
    best_check = np.array(best).flatten()

    plt.step(range(len(cs1_check)), cs1_check, label='cs1 (UE1 connected?)')
    plt.step(range(len(best_check)), best_check, label='best gNB')
    plt.ylim(-0.2, 1.2);
    plt.legend();
    plt.show()
    eng.quit()

if __name__ == '__main__':
    main()

