f = open('./gaussian_test.log.final.opt.xyz')
EDA_input = open('./qchem_test.inp','w')
coord = f.readlines()
natom = int(coord[0].split()[0])/2
natom = int(natom)
EDA_input.write('$molecule\n')
EDA_input.write('0 1\n')
for i in range(2):
    EDA_input.write('--\n')
    EDA_input.write('0 1\n')
    for j in range(natom):
        EDA_input.write(coord[j+2+i*natom])
EDA_input.write('$end\n')
EDA_input.write('\n')
EDA_input.write('$rem\n')
EDA_input.write('{0:3}{1:18}{2:10}\n'.format('','JOBTYPE','eda'))
EDA_input.write('{0:3}{1:18}{2:10}\n'.format('','EDA2','2'))
EDA_input.write('{0:3}{1:18}{2:10}\n'.format('','METHOD','wB97x-D'))
EDA_input.write('{0:3}{1:18}{2:10}\n'.format('','BASIS','def2-SV(P)'))
EDA_input.write('{0:3}{1:18}{2:10}\n'.format('','SCF_CONVERGENCE','8'))
EDA_input.write('{0:3}{1:18}{2:10}\n'.format('','THRESH','14'))
EDA_input.write('{0:3}{1:18}{2:10}\n'.format('','SYMMETRY','false'))
EDA_input.write('{0:3}{1:18}{2:10}\n'.format('','DISP_FREE_X','wB97X-D'))
EDA_input.write('{0:3}{1:18}{2:10}\n'.format('','DISP_FREE_C','none'))
EDA_input.write('{0:3}{1:18}{2:10}\n'.format('','EDA_BSSE','true'))
EDA_input.write('$end')
EDA_input.close()
