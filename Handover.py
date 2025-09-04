import matlab.engine
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PROJECT_DIR = Path(r"C:\Users\ASUS\Desktop\硕士毕设\matlab代码\Topo_5Gsimulink_handover.slx").parent
MODEL = 'Topo_5Gsimulink_handover'


def run_handover_simulation():
    """Starting 5G handover scenario simulation..."""
    eng = matlab.engine.start_matlab()
    eng.cd(str(PROJECT_DIR), nargout=0)
    eng.load_system(MODEL, nargout=0)

    print("Starting 5G handover scenario simulation...")
    print("=" * 50)

    # Define handover simulation scenarios - run complete simulation with two phases
    total_duration = 8  # Total simulation time: 8 seconds

    scenario_config = {
        'phase1': {
            'name': 'UE covered by gNB1',
            'amp': 30,  # gNB1 strong signal
            'amp2': 20,  # gNB2 weaker signal
            'start_time': 0,
            'end_time': 3,
            'description': 'UE connected to gNB1 due to stronger signal'
        },
        'phase2': {
            'name': 'UE movement triggered handover',
            'amp': 10,  # gNB1 signal weakened significantly
            'amp2': 25,  # gNB2 signal remains strong
            'start_time': 3,
            'end_time': 8,
            'description': 'gNB1 signal weakens, gNB2 becomes better choice'
        }
    }

    try:
        print(f"\nSetting complete simulation parameters (Total duration: {total_duration}s)")

        # Clear workspace
        eng.evalin('base', 'clear cs1 cs2 req1 req2 best AMF HDctrl_check gNB_request_check2', nargout=0)

        # Set initial parameters - start with phase1 parameters
        initial_config = scenario_config['phase1']
        eng.workspace['amp'] = float(initial_config['amp'])
        eng.workspace['amp2'] = float(initial_config['amp2'])
        eng.workspace['amf_state'] = 1.0
        eng.workspace['gnb1_status'] = 1.0
        eng.workspace['gnb2_status'] = 1.0

        # Set total simulation time
        eng.set_param(MODEL, 'StopTime', str(total_duration), nargout=0)

        # Directly set block parameters
        try:
            eng.set_param(f'{MODEL}/rsrp1', 'Amplitude', str(initial_config['amp']), nargout=0)
            print(f"    Set rsrp1 initial Amplitude = {initial_config['amp']}")
        except Exception as e:
            print(f"    Failed to set rsrp1 parameter: {e}")

        try:
            eng.set_param(f'{MODEL}/rsrp2', 'Amplitude', str(initial_config['amp2']), nargout=0)
            print(f"    Set rsrp2 initial Amplitude = {initial_config['amp2']}")
        except Exception as e:
            print(f"    Failed to set rsrp2 parameter: {e}")

        print(f"    Initial MATLAB workspace parameters:")
        print(f"    amp (gNB1) = {eng.workspace['amp']}")
        print(f"    amp2 (gNB2) = {eng.workspace['amp2']}")

        # Update model
        eng.set_param(MODEL, 'SimulationCommand', 'update', nargout=0)

        print(f"\nStarting complete simulation ({total_duration}s)...")
        print("Note: Signal strength will change at 3rd second during simulation")

        # Run complete simulation
        eng.sim(MODEL, nargout=0)
        print(f"    Complete simulation finished")

        # Check output variables
        workspace_vars = eng.who()
        print(f"    Workspace variables: {workspace_vars}")

        # Read simulation results - using more reliable method
        signals = {}
        signal_names = ['cs1', 'cs2', 'req1', 'req2', 'best', 'AMF', 'HDctrl_check', 'gNB_request_check2']

        for var_name in signal_names:
            if var_name not in workspace_vars:
                print(f"    Warning: Variable {var_name} not found in workspace")
                continue

            try:
                # Get variable
                var_data = eng.workspace[var_name]

                # Check data type and structure
                print(f"    Processing variable {var_name}...")

                if isinstance(var_data, dict):
                    # Simulink To Workspace struct format
                    if 'signals' in var_data:
                        if isinstance(var_data['signals'], dict) and 'values' in var_data['signals']:
                            # Standard struct format: var.signals.values
                            data_values = var_data['signals']['values']
                            if hasattr(data_values, '__len__') and len(data_values) > 0:
                                signals[var_name] = np.array(data_values).flatten()
                            else:
                                signals[var_name] = np.array([data_values]).flatten()
                        else:
                            print(f"      {var_name}: signals field format incorrect")
                            continue
                    elif 'Data' in var_data:
                        # Another possible struct format
                        signals[var_name] = np.array(var_data['Data']).flatten()
                    else:
                        print(f"      {var_name}: Unknown struct format, keys: {var_data.keys()}")
                        continue
                elif hasattr(var_data, '__len__'):
                    # Direct array format
                    signals[var_name] = np.array(var_data).flatten()
                else:
                    # Scalar value
                    signals[var_name] = np.array([var_data])

                print(f"      Successfully read {var_name}, length: {len(signals[var_name])}")
                print(f"      {var_name} first 5 values: {signals[var_name][:min(5, len(signals[var_name]))]}")
                print(
                    f"      {var_name} last 5 values: {signals[var_name][-min(5, len(signals[var_name])):] if len(signals[var_name]) > 5 else 'N/A'}")

            except Exception as e:
                print(f"      Failed to read {var_name}: {e}")

                # Backup method: use eval
                try:
                    # Try different access methods
                    test_commands = [
                        f'{var_name}.signals.values',
                        f'{var_name}.Data',
                        f'{var_name}'
                    ]

                    for cmd in test_commands:
                        try:
                            result = eng.eval(cmd)
                            signals[var_name] = np.array(result).flatten()
                            print(f"      Backup method successfully read {var_name} using: {cmd}")
                            break
                        except:
                            continue

                    if var_name not in signals:
                        print(f"      All methods failed for: {var_name}")

                except Exception as e2:
                    print(f"      Backup method also failed: {e2}")

        # Check required signals
        required_signals = ['cs1', 'cs2', 'best']
        missing_signals = [sig for sig in required_signals if sig not in signals]

        if missing_signals:
            print(f"Error: Missing critical signals: {missing_signals}")
            eng.quit()
            return None

        # Get time axis - try to get actual time from simulation
        try:
            # Method 1: Get time from To Workspace
            if 'cs1' in workspace_vars:
                cs1_data = eng.workspace['cs1']
                if isinstance(cs1_data, dict) and 'time' in cs1_data:
                    time_data = np.array(cs1_data['time']).flatten()
                    print(f"    Got time axis from cs1 struct, length: {len(time_data)}")
                else:
                    # Method 2: Generate time axis based on data length and simulation time
                    data_length = len(signals['cs1'])
                    time_data = np.linspace(0, total_duration, data_length)
                    print(f"    Generated linear time axis, length: {len(time_data)}")
            else:
                data_length = len(signals['cs1'])
                time_data = np.linspace(0, total_duration, data_length)
                print(f"    Using default linear time axis, length: {len(time_data)}")

        except Exception as e:
            print(f"    Failed to get time axis: {e}")
            data_length = len(signals['cs1']) if 'cs1' in signals else 1000
            time_data = np.linspace(0, total_duration, data_length)
            print(f"    Using backup time axis, length: {len(time_data)}")

        # Ensure all signals have consistent length
        min_length = min(len(signals[name]) for name in signals.keys())
        time_data = time_data[:min_length] if len(time_data) > min_length else time_data

        for name in signals.keys():
            if len(signals[name]) > min_length:
                signals[name] = signals[name][:min_length]

        # Add time axis to signals dictionary
        signals['time'] = time_data

        print(f"\nData summary:")
        print(f"  Time range: {time_data[0]:.3f} - {time_data[-1]:.3f} seconds")
        print(f"  Data points: {len(time_data)}")

        # Analyze data at key time points
        phase1_end_idx = np.argmin(np.abs(time_data - 3.0))  # 3rd second
        print(f"\nKey time point analysis:")
        print(f"  Around 3rd second (index {phase1_end_idx}):")
        if 'cs1' in signals and 'cs2' in signals:
            print(f"    cs1: {signals['cs1'][max(0, phase1_end_idx - 2):phase1_end_idx + 3]}")
            print(f"    cs2: {signals['cs2'][max(0, phase1_end_idx - 2):phase1_end_idx + 3]}")
        if 'best' in signals:
            print(f"    best: {signals['best'][max(0, phase1_end_idx - 2):phase1_end_idx + 3]}")

        # Check data at final moment
        print(f"  Final moment:")
        if 'cs1' in signals and 'cs2' in signals:
            print(f"    cs1: {signals['cs1'][-5:]}")
            print(f"    cs2: {signals['cs2'][-5:]}")
        if 'best' in signals:
            print(f"    best: {signals['best'][-5:]}")

        eng.quit()
        return signals

    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        eng.quit()
        return None


def analyze_handover_performance(signals):
    """Analyze handover performance"""
    if not signals or 'time' not in signals:
        print("Error: No valid simulation data for analysis")
        return None

    print("\n" + "=" * 50)
    print("        HANDOVER PERFORMANCE ANALYSIS")
    print("=" * 50)

    time_data = signals['time']
    cs1 = signals.get('cs1', np.zeros_like(time_data))
    cs2 = signals.get('cs2', np.zeros_like(time_data))
    best = signals.get('best', np.zeros_like(time_data))

    # Key time point
    handover_trigger_time = 3.0  # Handover starts at 3rd second

    # Find trigger time index
    trigger_idx = np.argmin(np.abs(time_data - handover_trigger_time))

    # 1. Analyze connection status before and after handover
    print("Connection status analysis before and after handover:")

    # Phase 1 statistics (0-3s)
    phase1_mask = time_data <= 3.0
    if np.any(phase1_mask):
        phase1_gnb1_rate = np.mean(cs1[phase1_mask]) * 100
        phase1_gnb2_rate = np.mean(cs2[phase1_mask]) * 100
        print(
            f"  Phase 1 (0-3s): gNB1 connection rate={phase1_gnb1_rate:.1f}%, gNB2 connection rate={phase1_gnb2_rate:.1f}%")

    # Phase 2 statistics (3-8s)
    phase2_mask = time_data > 3.0
    if np.any(phase2_mask):
        phase2_gnb1_rate = np.mean(cs1[phase2_mask]) * 100
        phase2_gnb2_rate = np.mean(cs2[phase2_mask]) * 100
        print(
            f"  Phase 2 (3-8s): gNB1 connection rate={phase2_gnb1_rate:.1f}%, gNB2 connection rate={phase2_gnb2_rate:.1f}%")

    # 2. Handover delay analysis
    handover_delay = calculate_handover_delay(signals, handover_trigger_time)

    # 3. Service interruption analysis
    service_interruption = calculate_service_interruption(signals)

    # 4. Handover success rate analysis
    handover_success = analyze_handover_success(signals, handover_trigger_time)

    print(f"\nPerformance metrics:")
    print(f"  Handover delay:        {handover_delay:.3f} seconds" if handover_delay != float(
        'inf') else "  Handover delay:        No handover detected")
    print(f"  Service interruption:  {service_interruption:.3f} seconds")
    print(f"  Handover success:      {'Yes' if handover_success else 'No'}")

    return {
        'handover_delay': handover_delay,
        'service_interruption': service_interruption,
        'handover_success': handover_success,
        'phase1_stats': {
            'gnb1_rate': phase1_gnb1_rate if 'phase1_gnb1_rate' in locals() else 0,
            'gnb2_rate': phase1_gnb2_rate if 'phase1_gnb2_rate' in locals() else 0
        },
        'phase2_stats': {
            'gnb1_rate': phase2_gnb1_rate if 'phase2_gnb1_rate' in locals() else 0,
            'gnb2_rate': phase2_gnb2_rate if 'phase2_gnb2_rate' in locals() else 0
        },
        'signals': signals
    }


def calculate_handover_delay(signals, trigger_time):
    """Calculate handover delay"""
    time_data = signals['time']
    cs1 = signals.get('cs1', np.zeros_like(time_data))
    cs2 = signals.get('cs2', np.zeros_like(time_data))

    trigger_idx = np.argmin(np.abs(time_data - trigger_time))

    # Find handover completion point: from gNB1(cs1=1) to gNB2(cs2=1)
    for i in range(trigger_idx, len(cs1)):
        if cs1[i] == 0 and cs2[i] == 1:  # Handover to gNB2
            return time_data[i] - trigger_time

    return float('inf')  # Handover not completed


def calculate_service_interruption(signals):
    """Calculate total service interruption time"""
    time_data = signals['time']
    cs1 = signals.get('cs1', np.zeros_like(time_data))
    cs2 = signals.get('cs2', np.zeros_like(time_data))

    if len(time_data) <= 1:
        return 0

    interruption_total = 0

    for i in range(1, len(cs1)):
        if cs1[i] == 0 and cs2[i] == 0:  # No service state
            time_diff = time_data[i] - time_data[i - 1]
            interruption_total += time_diff

    return interruption_total


def analyze_handover_success(signals, trigger_time):
    """Analyze whether handover is successful"""
    time_data = signals['time']
    cs1 = signals.get('cs1', np.zeros_like(time_data))
    cs2 = signals.get('cs2', np.zeros_like(time_data))

    # Check main connection before and after handover
    pre_switch_mask = time_data < trigger_time
    post_switch_mask = time_data >= trigger_time + 1  # 1 second buffer time

    if not np.any(pre_switch_mask) or not np.any(post_switch_mask):
        return False

    # Before handover: mainly connected to gNB1, after handover: mainly connected to gNB2
    pre_mainly_gnb1 = np.mean(cs1[pre_switch_mask]) > 0.5
    post_mainly_gnb2 = np.mean(cs2[post_switch_mask]) > 0.5

    return pre_mainly_gnb1 and post_mainly_gnb2


def plot_handover_results(analysis):
    """Plot handover simulation results"""
    signals = analysis['signals']

    if 'time' not in signals:
        print("Error: No time data available for plotting")
        return

    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    fig.suptitle('5G Network Handover Simulation Results Analysis', fontsize=16, fontweight='bold')

    # Set font for better display
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    time_axis = signals['time']

    # 1. UE connection status
    cs1 = signals.get('cs1', np.zeros_like(time_axis))
    cs2 = signals.get('cs2', np.zeros_like(time_axis))

    axes[0].step(time_axis, cs1, 'b-', linewidth=2, label='UE-gNB1 Connection')
    axes[0].step(time_axis, cs2, 'r-', linewidth=2, label='UE-gNB2 Connection')

    # Add phase annotations
    add_phase_annotations(axes[0])

    axes[0].set_ylabel('Connection Status')
    axes[0].set_title('UE Connection Status Change')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.1, 1.1)

    # 2. Base station request status
    req1 = signals.get('req1', np.zeros_like(time_axis))
    req2 = signals.get('req2', np.zeros_like(time_axis))

    axes[1].step(time_axis, req1, 'b-', linewidth=2, label='gNB1→AMF Request')
    axes[1].step(time_axis, req2, 'r-', linewidth=2, label='gNB2→AMF Request')

    add_phase_annotations(axes[1])

    axes[1].set_ylabel('Request Status')
    axes[1].set_title('Base Station to AMF Request Status')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.1, 1.1)

    # 3. AMF response status
    amf_resp = signals.get('AMF', np.zeros_like(time_axis))

    axes[2].step(time_axis, amf_resp, 'purple', linewidth=2, label='AMF Response')

    add_phase_annotations(axes[2])

    axes[2].set_ylabel('AMF Status')
    axes[2].set_title('AMF Response Status')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(-0.1, 1.1)

    # 4. Best base station selection
    best = signals.get('best', np.zeros_like(time_axis))

    axes[3].step(time_axis, best, 'orange', linewidth=2, label='Best gNB (0=gNB1, 1=gNB2)')

    add_phase_annotations(axes[3])

    axes[3].set_ylabel('Best gNB')
    axes[3].set_xlabel('Time (seconds)')
    axes[3].set_title('Best Base Station Selection')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    axes[3].set_ylim(-0.1, 1.1)

    plt.tight_layout()
    plt.show()

    # Plot performance comparison
    plot_performance_comparison(analysis)


def add_phase_annotations(ax):
    """Add phase annotations to charts"""
    colors = ['lightblue', 'lightyellow']
    labels = ['Phase 1: UE@gNB1', 'Phase 2: Signal Weakening Handover']
    durations = [3, 5]  # Phase durations

    current_time = 0
    for color, label, duration in zip(colors, labels, durations):
        rect = Rectangle((current_time, -0.05), duration, 1.1,
                         facecolor=color, alpha=0.3, edgecolor='none')
        ax.add_patch(rect)

        # Add text annotation
        ax.text(current_time + duration / 2, 0.95, label,
                ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

        current_time += duration


def plot_performance_comparison(analysis):
    """Plot performance comparison charts"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Handover Performance Metrics Comparison', fontsize=14, fontweight='bold')

    # Delay metrics comparison
    delays = [
        analysis['handover_delay'] if analysis['handover_delay'] != float('inf') else 0,
        analysis['service_interruption']
    ]
    delay_labels = ['Handover Delay', 'Service Interruption']

    bars1 = ax1.bar(delay_labels, delays, color=['skyblue', 'salmon'])
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Delay Performance Metrics')
    ax1.grid(True, alpha=0.3)

    # Display values on bars
    for bar, delay in zip(bars1, delays):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + max(delays) * 0.01,
                 f'{height:.3f}s', ha='center', va='bottom')

    # Connection success rate comparison by phase
    phase1_stats = analysis['phase1_stats']
    phase2_stats = analysis['phase2_stats']

    gnb1_rates = [phase1_stats['gnb1_rate'], phase2_stats['gnb1_rate']]
    gnb2_rates = [phase1_stats['gnb2_rate'], phase2_stats['gnb2_rate']]

    x = np.arange(2)  # Two phases
    width = 0.35

    bars2 = ax2.bar(x - width / 2, gnb1_rates, width, label='gNB1 Connection Rate', color='blue', alpha=0.7)
    bars3 = ax2.bar(x + width / 2, gnb2_rates, width, label='gNB2 Connection Rate', color='red', alpha=0.7)

    ax2.set_ylabel('Connection Success Rate (%)')
    ax2.set_title('Connection Success Rate by Phase')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Phase 1', 'Phase 2'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)

    # Display values on bars
    for bar, rate in zip(bars2, gnb1_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom')

    for bar, rate in zip(bars3, gnb2_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def main():
    """Main program"""
    try:
        # Run handover simulation
        signals = run_handover_simulation()

        if not signals:
            print("Simulation failed, no result data obtained")
            print("\nPossible causes:")
            print("1. Simulink model file path is incorrect")
            print("2. Missing To Workspace blocks in model or variable name mismatch")
            print("3. Error occurred during simulation execution")
            print("4. MATLAB engine connection issues")
            print("5. Model requires dynamic parameter changes during simulation")
            return

        print(f"\nSuccessfully obtained complete simulation data")

        # Analyze performance
        analysis = analyze_handover_performance(signals)

        if analysis is None:
            print("Performance analysis failed")
            return

        # Plot results
        plot_handover_results(analysis)

        print("\n5G handover simulation analysis completed!")

        # Output summary
        print("\n" + "=" * 50)
        print("Simulation Summary:")
        print(f"  Handover: {'Successful' if analysis['handover_success'] else 'Failed'}")
        print(f"  Handover delay: {analysis['handover_delay']:.3f}s" if analysis['handover_delay'] != float(
            'inf') else "  Handover delay: Not completed")
        print(f"  Service interruption: {analysis['service_interruption']:.3f}s")

    except Exception as e:
        print(f"Program execution error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()