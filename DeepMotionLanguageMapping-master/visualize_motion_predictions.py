import argparse
import cPickle as pickle
import os
from tempfile import mkstemp
import subprocess

import matplotlib.pyplot as plt
import numpy as np


def find_package(package_name):
    cmd = ['cmake', '--find-package', '-DNAME=' + package_name, '-DLANGUAGE=CXX',  '-DCOMPILER_ID=GNU', '-DMODE=COMPILE']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if len(err) > 0:
        raise RuntimeError(err)
    out = out.strip()
    if out.find(package_name + ' not found.') != -1:
        return None
    ls = [x.strip() for x in out.split('-I') if len(x.strip()) > 0]
    package_path = None
    if len(ls) == 0:
        # Package seems to exist but we can't parse it properly. Fall back
        # to manually coded options.
        options = ['/Users/matze/Labs', '/home/plappert/Code/']
        for option in options:
            path = os.path.join(option, package_name)
            if os.path.exists(path):
                package_path = path
                break
    else:
        package_path = ls[0]
    return package_path


def plot_data(ax, data, title=None):
    heatmap = ax.pcolor(np.swapaxes(data, 0, 1), vmin=-1., vmax=1.)

    # Configure y axis.
    ax.set_ylim([0, data.shape[1]])
    ax.set_ylabel('Features')

    # Configure x axis.
    ax.set_xlabel('Time')
    ax.set_xlim([0, data.shape[0]])


def get_mmm_xml_representation(motion, joint_names, model_path, step_duration=.1):
    import pymmm
    nb_joints = len(joint_names)
    motion = motion.astype('float32')
    assert motion.ndim == 2
    assert motion.shape[-1] == nb_joints

    mmm_motion = pymmm.Motion('export')
    mmm_motion.setJointOrder(joint_names)
    for idx, frame in enumerate(motion):
        mmm_frame = pymmm.MotionFrame(nb_joints)
        mmm_frame.timestep = float(idx) * step_duration
        mmm_frame.joint = frame.astype('float32')
        mmm_frame.setRootPos(np.array([0., 0., 0.], dtype='float32'))
        mmm_frame.setRootRot(np.array([0., 0., 0.], dtype='float32'))
        mmm_motion.addMotionFrame(mmm_frame)
    
    inner_xml = mmm_motion.toXML()
    xml_model = '\t\t<Model>\n\t\t\t<File>{}</File>\n\t\t</Model>'.format(model_path)
    inner_xml = inner_xml.replace('<Motion name=\'export\'>', '<Motion name=\'export\'>\n' + xml_model)
    xml = '<?xml version=\'1.0\'?>\n<MMM>\n{}</MMM>\n'.format(inner_xml)
    return xml


def main(args):
    # Find necessary MMM files.
    mmm_root_path = find_package('MMMTools')
    if not mmm_root_path:
        exit('could not find MMMTools')
    model_path = os.path.join(mmm_root_path, 'data', 'Model', 'Winter', 'mmm.xml')
    bin_path = os.path.join(mmm_root_path, 'build', 'bin', 'MMMViewer')

    # Load predictions.
    with open(args.input, 'rb') as f:
        d = pickle.load(f)
        joint_names = d['joint_names']
        scaler = pickle.loads(d['scaler'])
        data = d['decoded_data']
    if args.idx is None:
        sample_idx = np.random.randint(len(data))
    else:
        sample_idx = args.idx
    
    # Visualize data, i.e. plot motion and display text.
    print('Visualizing index {} from "{}" ...'.format(sample_idx, args.input))
    best_idx = np.argmax(data[sample_idx]['log_probabilities'])
    hypothesis = data[sample_idx]['hypotheses'][best_idx]
    reference = data[sample_idx]['references'][0]
    language = data[sample_idx]['language']
    log_probability = data[sample_idx]['log_probabilities'][best_idx]
    print('  language: {}'.format(language))
    print('  log_probabilities {}'.format(log_probability))
    delta_steps = reference.shape[0] - hypothesis.shape[0]
    if delta_steps > 0:
        hypothesis = np.vstack([hypothesis, np.zeros((delta_steps, hypothesis.shape[1]))])
    elif delta_steps < 0:
        reference = np.vstack([reference, np.zeros((-delta_steps, reference.shape[1]))])
    assert hypothesis.shape == reference.shape
    fig, subplots = plt.subplots(nrows=2)
    for d, plot in zip([hypothesis, reference], subplots):
        plot_data(plot, d)
    plt.tight_layout()
    plt.show()
    print('done')

    # Export motion to MMM format.
    hypothesis_fd, hypothesis_path = mkstemp(suffix='_{}_hypothesis'.format(sample_idx))
    reference_fd, reference_path = mkstemp(suffix='_{}_reference'.format(sample_idx))
    nb_joints = len(joint_names)
    with os.fdopen(hypothesis_fd, 'w') as tmp:
        decodable_hypothesis = scaler.inverse_transform(hypothesis[:, :nb_joints])
        tmp.write(get_mmm_xml_representation(decodable_hypothesis, joint_names, model_path))
    with os.fdopen(reference_fd, 'w') as tmp:
        decodable_reference = scaler.inverse_transform(reference[:, :nb_joints])
        tmp.write(get_mmm_xml_representation(decodable_reference, joint_names, model_path))

    # Visualize motion using MMM toolkit.
    print('visualizing hypothesis "{}"'.format(hypothesis_path))
    os.system(bin_path + ' --motion {} > /dev/null &'.format(hypothesis_path))
    print('visualizing reference "{}"'.format(reference_path))
    os.system(bin_path + ' --motion {} > /dev/null'.format(reference_path))

    # Clean up.
    os.remove(hypothesis_path)
    os.remove(reference_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input', type=str)
    parser.add_argument('--idx', type=int, default=None)
    main(parser.parse_args())
