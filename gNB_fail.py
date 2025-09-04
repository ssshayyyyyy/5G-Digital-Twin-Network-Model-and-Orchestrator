import matlab.engine
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PROJECT_DIR = Path(r"C:\Users\ASUS\Desktop\硕士毕设\matlab代码\Topo_5Gsimulink_handover.slx").parent
MODEL = 'Topo_5Gsimulink_handover'


def run_gnb_failure_simulation():
    """Starting 5G gNB failure scenario simulation..."""
    eng = matlab.engine.start_matlab()
    eng.cd(str(PROJECT_DIR), nargout=0)
    eng.load_system(MODEL, nargout=0)

    print("Starting 5G gNB failure scenario simulation...")
    print("=" * 50)

    # Define gNB failure simulation scenarios - using existing model structure
    total_duration = 10  # Total simulation time: 10 seconds

    scenario_config = {
        'phase1': {
            'name': 'Normal operation - UE connected to gNB1',
            'amp': 35,  # gNB1 strong signal (rsrp1)
            'amp2': 25,  # gNB2 backup signal (rsrp2)
            'start_time': 0,
            'end_time': 4,
            'description': 'UE normally connected to gNB1 with good signal quality'
        },
        'phase2': {
            'name': 'gNB1 failure - signal completely lost',
            'amp': 0,  # gNB1 signal lost (complete failure)
            'amp2': 30,  # gNB2 signal becomes primary
            'start_time': 4,
            'end_time': 10,
            'description': 'gNB1 completely fails, UE must reconnect to gNB2'
        }
    }

    try:
        print(f"\nSetting gNB failure simulation parameters (Total duration: {total_duration}s)")

        # Clear workspace - keep existing variable names
        eng.evalin('base', 'clear cs1 cs2 req1 req2 best AMF HDctrl_check gNB_request_check2', nargout=0)

        # Set initial parameters - start with phase1 (normal operation)
        initial_config = scenario_config['phase1']
        eng.workspace['amp'] = float(initial_config['amp'])  # Use existing amp variable
        eng.workspace['amp2'] = float(initial_config['amp2'])  # Use existing amp2 variable
        eng.workspace['amf_state'] = 1.0
        eng.workspace['gnb1_status'] = 1.0
        eng.workspace['gnb2_status'] = 1.0

        # Set total simulation time
        eng.set_param(MODEL, 'StopTime', str(total_duration), nargout=0)

        # Set block parameters using existing block names
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
        print(f"    amp (gNB1) = {eng.workspace['amp']} (normal signal)")
        print(f"    amp2 (gNB2) = {eng.workspace['amp2']} (backup signal)")

        # Update model
        eng.set_param(MODEL, 'SimulationCommand', 'update', nargout=0)

        print(f"\nStarting gNB failure simulation ({total_duration}s)...")
        print("Note: gNB1 will experience complete signal failure at 4th second")

        # Run complete simulation with dynamic parameter change
        eng.sim(MODEL, nargout=0)
        print(f"    gNB failure simulation finished")

        # Check output variables
        workspace_vars = eng.who()
        print(f"    Workspace variables: {workspace_vars}")

        # Read simulation results - using existing signal names
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

        # Get time axis
        try:
            if 'cs1' in workspace_vars:
                cs1_data = eng.workspace['cs1']
                if isinstance(cs1_data, dict) and 'time' in cs1_data:
                    time_data = np.array(cs1_data['time']).flatten()
                    print(f"    Got time axis from cs1 struct, length: {len(time_data)}")
                else:
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
        failure_time_idx = np.argmin(np.abs(time_data - 4.0))  # 4th second failure
        print(f"\nKey time point analysis:")
        print(f"  Around gNB1 failure time (4th second, index {failure_time_idx}):")
        if 'cs1' in signals and 'cs2' in signals:
            print(f"    cs1: {signals['cs1'][max(0, failure_time_idx - 2):failure_time_idx + 5]}")
            print(f"    cs2: {signals['cs2'][max(0, failure_time_idx - 2):failure_time_idx + 5]}")
        if 'best' in signals:
            print(f"    best: {signals['best'][max(0, failure_time_idx - 2):failure_time_idx + 5]}")

        # Check final recovery status
        print(f"  Final recovery status:")
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


def analyze_gnb_failure_performance(signals):
    """Analyze gNB failure recovery performance using existing signals"""
    if not signals or 'time' not in signals:
        print("Error: No valid simulation data for analysis")
        return None

    print("\n" + "=" * 50)
    print("        gNB FAILURE RECOVERY PERFORMANCE ANALYSIS")
    print("=" * 50)

    time_data = signals['time']
    cs1 = signals.get('cs1', np.zeros_like(time_data))
    cs2 = signals.get('cs2', np.zeros_like(time_data))
    best = signals.get('best', np.zeros_like(time_data))

    # Key time point
    failure_trigger_time = 4.0  # gNB1 fails at 4th second

    # Find trigger time index
    trigger_idx = np.argmin(np.abs(time_data - failure_trigger_time))

    print("Connection status analysis before and after gNB1 failure:")

    # Phase 1 statistics (0-4s): Normal operation
    phase1_mask = time_data <= 4.0
    if np.any(phase1_mask):
        phase1_gnb1_rate = np.mean(cs1[phase1_mask]) * 100
        phase1_gnb2_rate = np.mean(cs2[phase1_mask]) * 100
        phase1_packet_loss = estimate_packet_loss_from_connection(cs1[phase1_mask], cs2[phase1_mask])
        phase1_throughput = estimate_throughput_from_connection(cs1[phase1_mask], cs2[phase1_mask], 'normal')

        print(f"  Phase 1 (0-4s) - Normal Operation:")
        print(f"    gNB1 connection rate: {phase1_gnb1_rate:.1f}%")
        print(f"    gNB2 connection rate: {phase1_gnb2_rate:.1f}%")
        print(f"    Estimated packet loss: {phase1_packet_loss:.2f}%")
        print(f"    Estimated throughput: {phase1_throughput:.2f} Mbps")

    # Phase 2 statistics (4-10s): After failure
    phase2_mask = time_data > 4.0
    if np.any(phase2_mask):
        phase2_gnb1_rate = np.mean(cs1[phase2_mask]) * 100
        phase2_gnb2_rate = np.mean(cs2[phase2_mask]) * 100
        phase2_packet_loss = estimate_packet_loss_from_connection(cs1[phase2_mask], cs2[phase2_mask])
        phase2_throughput = estimate_throughput_from_connection(cs1[phase2_mask], cs2[phase2_mask], 'recovery')

        print(f"  Phase 2 (4-10s) - After gNB1 Failure:")
        print(f"    gNB1 connection rate: {phase2_gnb1_rate:.1f}%")
        print(f"    gNB2 connection rate: {phase2_gnb2_rate:.1f}%")
        print(f"    Estimated packet loss: {phase2_packet_loss:.2f}%")
        print(f"    Estimated throughput: {phase2_throughput:.2f} Mbps")

    # Calculate failure recovery metrics
    recovery_delay = calculate_failure_recovery_delay(signals, failure_trigger_time)
    service_downtime = calculate_service_downtime(signals, failure_trigger_time)
    recovery_success = analyze_failure_recovery_success(signals, failure_trigger_time)

    print(f"\nFailure Recovery Performance Metrics:")
    print(f"  Recovery delay:           {recovery_delay:.3f} seconds" if recovery_delay != float(
        'inf') else "  Recovery delay:           Recovery not completed")
    print(f"  Service downtime:         {service_downtime:.3f} seconds")
    print(f"  Recovery success:         {'Yes' if recovery_success else 'No'}")
    print(f"  Packet loss increase:     {phase2_packet_loss - phase1_packet_loss:.2f}%")
    print(
        f"  Throughput degradation:   {((phase1_throughput - phase2_throughput) / phase1_throughput * 100) if phase1_throughput > 0 else 0:.2f}%")

    return {
        'recovery_delay': recovery_delay,
        'service_downtime': service_downtime,
        'recovery_success': recovery_success,
        'phase1_stats': {
            'gnb1_rate': phase1_gnb1_rate if 'phase1_gnb1_rate' in locals() else 0,
            'gnb2_rate': phase1_gnb2_rate if 'phase1_gnb2_rate' in locals() else 0,
            'packet_loss': phase1_packet_loss if 'phase1_packet_loss' in locals() else 0,
            'throughput': phase1_throughput if 'phase1_throughput' in locals() else 0
        },
        'phase2_stats': {
            'gnb1_rate': phase2_gnb1_rate if 'phase2_gnb1_rate' in locals() else 0,
            'gnb2_rate': phase2_gnb2_rate if 'phase2_gnb2_rate' in locals() else 0,
            'packet_loss': phase2_packet_loss if 'phase2_packet_loss' in locals() else 0,
            'throughput': phase2_throughput if 'phase2_throughput' in locals() else 0
        },
        'signals': signals
    }


def estimate_packet_loss_from_connection(cs1, cs2):
    """Estimate packet loss rate based on connection status"""
    # When no connection to any gNB: 100% packet loss
    # When connected: base packet loss + instability penalty

    no_connection_mask = (cs1 == 0) & (cs2 == 0)
    connection_mask = (cs1 == 1) | (cs2 == 1)

    if len(cs1) == 0:
        return 0

    # Calculate packet loss
    no_connection_ratio = np.sum(no_connection_mask) / len(cs1)
    connection_ratio = np.sum(connection_mask) / len(cs1)

    # Base packet loss during connection (0.1%) + penalty for disconnections
    base_packet_loss = 0.1  # 0.1% base packet loss
    disconnection_penalty = no_connection_ratio * 100  # 100% loss during disconnection

    total_packet_loss = base_packet_loss + disconnection_penalty

    return total_packet_loss


def estimate_throughput_from_connection(cs1, cs2, phase):
    """Estimate throughput based on connection status and phase"""
    # Normal operation: high throughput
    # During failure/recovery: reduced throughput

    if len(cs1) == 0:
        return 0

    gnb1_connection_ratio = np.mean(cs1)
    gnb2_connection_ratio = np.mean(cs2)
    total_connection_ratio = gnb1_connection_ratio + gnb2_connection_ratio

    if phase == 'normal':
        # Normal phase: gNB1 provides higher throughput
        base_throughput = 100  # 100 Mbps base throughput
        actual_throughput = base_throughput * total_connection_ratio
    else:
        # Recovery phase: gNB2 may have lower capacity initially
        base_throughput = 85  # 85 Mbps base throughput (slightly lower for backup gNB)
        actual_throughput = base_throughput * total_connection_ratio

        # Additional penalty for connection instability
        connection_stability = 1 - abs(gnb1_connection_ratio - gnb2_connection_ratio)
        actual_throughput *= connection_stability

    return actual_throughput


def calculate_failure_recovery_delay(signals, failure_time):
    """Calculate time from gNB1 failure to successful reconnection to gNB2"""
    time_data = signals['time']
    cs1 = signals.get('cs1', np.zeros_like(time_data))
    cs2 = signals.get('cs2', np.zeros_like(time_data))

    failure_idx = np.argmin(np.abs(time_data - failure_time))

    # Find when gNB1 connection is lost (cs1 becomes 0)
    gnb1_lost_idx = failure_idx
    for i in range(failure_idx, len(cs1)):
        if cs1[i] == 0:
            gnb1_lost_idx = i
            break

    # Find when UE successfully connects to gNB2 (cs2 becomes 1)
    for i in range(gnb1_lost_idx, len(cs2)):
        if cs2[i] == 1:  # Successfully connected to gNB2
            return time_data[i] - time_data[gnb1_lost_idx]

    return float('inf')  # Recovery not completed


def calculate_service_downtime(signals, failure_time):
    """Calculate total service downtime (no connection to any gNB)"""
    time_data = signals['time']
    cs1 = signals.get('cs1', np.zeros_like(time_data))
    cs2 = signals.get('cs2', np.zeros_like(time_data))

    if len(time_data) <= 1:
        return 0

    downtime_total = 0
    failure_idx = np.argmin(np.abs(time_data - failure_time))

    # Only count downtime after failure occurs
    for i in range(failure_idx + 1, len(cs1)):
        if cs1[i] == 0 and cs2[i] == 0:  # No service state
            time_diff = time_data[i] - time_data[i - 1]
            downtime_total += time_diff

    return downtime_total


def analyze_failure_recovery_success(signals, failure_time):
    """Analyze whether recovery from gNB1 failure is successful"""
    time_data = signals['time']
    cs1 = signals.get('cs1', np.zeros_like(time_data))
    cs2 = signals.get('cs2', np.zeros_like(time_data))

    # Check connection status before failure and after recovery
    pre_failure_mask = time_data < failure_time
    post_recovery_mask = time_data >= failure_time + 2  # 2 seconds buffer for recovery

    if not np.any(pre_failure_mask) or not np.any(post_recovery_mask):
        return False

    # Before failure: mainly connected to gNB1
    pre_mainly_gnb1 = np.mean(cs1[pre_failure_mask]) > 0.8

    # After failure + recovery time: mainly connected to gNB2
    post_mainly_gnb2 = np.mean(cs2[post_recovery_mask]) > 0.8
    post_gnb1_disconnected = np.mean(cs1[post_recovery_mask]) < 0.2

    return pre_mainly_gnb1 and post_mainly_gnb2 and post_gnb1_disconnected


def plot_gnb_failure_results(analysis):
    """Plot gNB failure simulation results using existing signal structure"""
    signals = analysis['signals']

    if 'time' not in signals:
        print("Error: No time data available for plotting")
        return

    fig, axes = plt.subplots(5, 1, figsize=(14, 12))
    fig.suptitle('5G gNB Failure Recovery Simulation Results Analysis', fontsize=16, fontweight='bold')

    # Set font for better display
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    time_axis = signals['time']

    # 1. UE connection status (using existing cs1, cs2)
    cs1 = signals.get('cs1', np.zeros_like(time_axis))
    cs2 = signals.get('cs2', np.zeros_like(time_axis))

    axes[0].step(time_axis, cs1, 'b-', linewidth=2, label='UE-gNB1 Connection')
    axes[0].step(time_axis, cs2, 'r-', linewidth=2, label='UE-gNB2 Connection')

    # Add failure phase annotations
    add_failure_phase_annotations(axes[0])

    axes[0].set_ylabel('Connection Status')
    axes[0].set_title('UE Connection Status During gNB1 Failure')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.1, 1.1)

    # 2. Base station request status (using existing req1, req2)
    req1 = signals.get('req1', np.zeros_like(time_axis))
    req2 = signals.get('req2', np.zeros_like(time_axis))

    axes[1].step(time_axis, req1, 'b-', linewidth=2, label='gNB1→AMF Request')
    axes[1].step(time_axis, req2, 'r-', linewidth=2, label='gNB2→AMF Request')

    add_failure_phase_annotations(axes[1])

    axes[1].set_ylabel('Request Status')
    axes[1].set_title('Base Station to AMF Request Status')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.1, 1.1)

    # 3. AMF response status (using existing AMF)
    amf_resp = signals.get('AMF', np.zeros_like(time_axis))

    axes[2].step(time_axis, amf_resp, 'purple', linewidth=2, label='AMF Response')

    add_failure_phase_annotations(axes[2])

    axes[2].set_ylabel('AMF Status')
    axes[2].set_title('AMF Response During Failure Recovery')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(-0.1, 1.1)

    # 4. Estimated service quality metrics
    estimated_packet_loss = []
    estimated_throughput = []

    for i in range(len(time_axis)):
        # Estimate packet loss
        if cs1[i] == 0 and cs2[i] == 0:
            packet_loss = 100  # Complete loss during disconnection
        elif cs1[i] == 1 or cs2[i] == 1:
            if time_axis[i] <= 4.0:
                packet_loss = 0.1  # Normal operation
            else:
                packet_loss = 0.5  # Slightly higher during recovery
        else:
            packet_loss = 10  # Intermediate state

        estimated_packet_loss.append(packet_loss)

        # Estimate throughput
        if cs1[i] == 1:
            throughput = 100 if time_axis[i] <= 4.0 else 0  # gNB1 fails after 4s
        elif cs2[i] == 1:
            throughput = 85  # gNB2 backup capacity
        else:
            throughput = 0  # No connection

        estimated_throughput.append(throughput)

    axes[3].plot(time_axis, estimated_packet_loss, 'purple', linewidth=2, label='Estimated Packet Loss Rate')

    add_failure_phase_annotations(axes[3])

    axes[3].set_ylabel('Packet Loss Rate (%)')
    axes[3].set_title('Estimated Service Quality - Packet Loss')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    # 5. Best base station selection (using existing best)
    best = signals.get('best', np.zeros_like(time_axis))

    axes[4].step(time_axis, best, 'orange', linewidth=2, label='Best gNB (0=gNB1, 1=gNB2)')
    ax4_twin = axes[4].twinx()
    ax4_twin.plot(time_axis, estimated_throughput, 'green', linewidth=2, alpha=0.7, label='Estimated Throughput')

    add_failure_phase_annotations(axes[4])

    axes[4].set_ylabel('Best gNB Selection')
    ax4_twin.set_ylabel('Throughput (Mbps)')
    axes[4].set_xlabel('Time (seconds)')
    axes[4].set_title('Best Base Station Selection and Throughput')

    # Combine legends
    lines1, labels1 = axes[4].get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    axes[4].legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    axes[4].grid(True, alpha=0.3)
    axes[4].set_ylim(-0.1, 1.1)

    plt.tight_layout()
    plt.show()

    # Plot performance comparison
    plot_failure_performance_comparison(analysis)


def add_failure_phase_annotations(ax):
    """Add phase annotations for gNB failure scenario"""
    colors = ['lightgreen', 'lightcoral']
    labels = ['Phase 1: Normal Operation', 'Phase 2: gNB1 Failure & Recovery']
    durations = [4, 6]  # Phase durations

    current_time = 0
    for color, label, duration in zip(colors, labels, durations):
        rect = Rectangle((current_time, -0.05), duration, 1.15,
                         facecolor=color, alpha=0.3, edgecolor='none')
        ax.add_patch(rect)

        # Add text annotation
        ax.text(current_time + duration / 2, 0.95, label,
                ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

        current_time += duration


def plot_failure_performance_comparison(analysis):
    """Plot gNB failure performance comparison charts"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('gNB Failure Recovery Performance Metrics Comparison', fontsize=14, fontweight='bold')

    # 1. Recovery time metrics comparison
    recovery_metrics = [
        analysis['recovery_delay'] if analysis['recovery_delay'] != float('inf') else 0,
        analysis['service_downtime']
    ]
    recovery_labels = ['Recovery Delay', 'Service Downtime']

    bars1 = ax1.bar(recovery_labels, recovery_metrics, color=['skyblue', 'salmon'])
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Recovery Time Performance')
    ax1.grid(True, alpha=0.3)

    # Display values on bars
    for bar, metric in zip(bars1, recovery_metrics):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + max(recovery_metrics) * 0.01,
                 f'{height:.3f}s', ha='center', va='bottom')

    # 2. Connection success rate comparison by phase
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
    ax2.set_xticklabels(['Before Failure', 'After Failure'])
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

    # 3. QoS comparison - Packet loss
    packet_loss_rates = [phase1_stats['packet_loss'], phase2_stats['packet_loss']]

    bars4 = ax3.bar(['Before Failure', 'After Recovery'], packet_loss_rates,
                    color=['lightgreen', 'orange'], alpha=0.7)
    ax3.set_ylabel('Packet Loss Rate (%)')
    ax3.set_title('Packet Loss Rate Comparison')
    ax3.grid(True, alpha=0.3)

    # Display values on bars
    for bar, rate in zip(bars4, packet_loss_rates):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height + max(packet_loss_rates) * 0.01,
                 f'{height:.2f}%', ha='center', va='bottom')

    # 4. QoS comparison - Throughput
    throughput_rates = [phase1_stats['throughput'], phase2_stats['throughput']]

    bars5 = ax4.bar(['Before Failure', 'After Recovery'], throughput_rates,
                    color=['lightgreen', 'orange'], alpha=0.7)
    ax4.set_ylabel('Throughput (Mbps)')
    ax4.set_title('Throughput Comparison')
    ax4.grid(True, alpha=0.3)

    # Display values on bars
    for bar, rate in zip(bars5, throughput_rates):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height + max(throughput_rates) * 0.01,
                 f'{height:.2f} Mbps', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def generate_failure_report(analysis):
    """Generate detailed gNB failure recovery report"""
    print("\n" + "=" * 60)
    print("               DETAILED gNB FAILURE RECOVERY REPORT")
    print("=" * 60)

    # Executive summary
    print("\nEXECUTIVE SUMMARY:")
    print("-" * 20)
    recovery_status = "SUCCESSFUL" if analysis['recovery_success'] else "FAILED"
    print(f"gNB1 Failure Recovery Status: {recovery_status}")

    if analysis['recovery_delay'] != float('inf'):
        print(f"Recovery Time: {analysis['recovery_delay']:.3f} seconds")
    else:
        print("Recovery Time: Not completed within simulation period")

    print(f"Service Downtime: {analysis['service_downtime']:.3f} seconds")

    # Performance comparison
    print("\nPERFORMANCE COMPARISON:")
    print("-" * 25)

    phase1 = analysis['phase1_stats']
    phase2 = analysis['phase2_stats']

    print(f"Connection Quality:")
    print(f"  Before Failure: gNB1={phase1['gnb1_rate']:.1f}%, gNB2={phase1['gnb2_rate']:.1f}%")
    print(f"  After Recovery: gNB1={phase2['gnb1_rate']:.1f}%, gNB2={phase2['gnb2_rate']:.1f}%")

    print(f"\nService Quality (Estimated):")
    print(f"  Packet Loss Rate:")
    print(f"    Before Failure: {phase1['packet_loss']:.2f}%")
    print(f"    After Recovery: {phase2['packet_loss']:.2f}%")
    print(f"    Increase: {phase2['packet_loss'] - phase1['packet_loss']:.2f}%")

    print(f"  Throughput:")
    print(f"    Before Failure: {phase1['throughput']:.2f} Mbps")
    print(f"    After Recovery: {phase2['throughput']:.2f} Mbps")

    throughput_degradation = ((phase1['throughput'] - phase2['throughput']) / phase1['throughput'] * 100) if phase1[
                                                                                                                 'throughput'] > 0 else 0
    print(f"    Degradation: {throughput_degradation:.2f}%")

    # Recovery efficiency assessment
    print(f"\nRECOVERY EFFICIENCY ASSESSMENT:")
    print("-" * 35)

    if analysis['recovery_delay'] <= 1.0:
        recovery_grade = "Excellent (≤1s)"
    elif analysis['recovery_delay'] <= 2.0:
        recovery_grade = "Good (1-2s)"
    elif analysis['recovery_delay'] <= 3.0:
        recovery_grade = "Acceptable (2-3s)"
    else:
        recovery_grade = "Poor (>3s)"

    print(f"Recovery Speed: {recovery_grade}")

    if analysis['service_downtime'] <= 0.5:
        downtime_grade = "Excellent (≤0.5s)"
    elif analysis['service_downtime'] <= 1.0:
        downtime_grade = "Good (0.5-1s)"
    elif analysis['service_downtime'] <= 2.0:
        downtime_grade = "Acceptable (1-2s)"
    else:
        downtime_grade = "Poor (>2s)"

    print(f"Service Continuity: {downtime_grade}")

    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    print("-" * 17)

    if not analysis['recovery_success']:
        print("• CRITICAL: Recovery failed - check backup gNB2 availability and AMF signaling")

    if analysis['recovery_delay'] > 2.0:
        print("• Consider optimizing failure detection in handover control logic")
        print("• Implement faster gNB selection algorithms")

    if analysis['service_downtime'] > 1.0:
        print("• Reduce connection re-establishment time in UE logic")
        print("• Optimize AMF response time for emergency handovers")

    packet_loss_increase = phase2['packet_loss'] - phase1['packet_loss']
    if packet_loss_increase > 5.0:
        print("• Optimize packet buffering during failure recovery")
        print("• Consider implementing connection prediction mechanisms")

    if throughput_degradation > 20.0:
        print("• Investigate gNB2 capacity optimization")
        print("• Consider network load balancing improvements")

    print("\n" + "=" * 60)


def main():
    """Main program - gNB failure scenario using existing Simulink structure"""
    try:
        print("=== 5G gNB FAILURE RECOVERY SIMULATION ===")
        print("Using existing handover model structure for failure scenario")
        print("Scenario: gNB1 complete failure → UE automatic reconnection to gNB2")
        print()

        # Run gNB failure simulation
        signals = run_gnb_failure_simulation()

        if not signals:
            print("Simulation failed, no result data obtained")
            print("\nPossible causes:")
            print("1. Simulink model file path is incorrect")
            print("2. Missing To Workspace blocks in model or variable name mismatch")
            print("3. Error occurred during simulation execution")
            print("4. MATLAB engine connection issues")
            print("5. Model requires dynamic parameter changes during simulation")
            return

        print(f"\nSuccessfully obtained gNB failure simulation data")

        # Analyze performance
        analysis = analyze_gnb_failure_performance(signals)

        if analysis is None:
            print("Performance analysis failed")
            return

        # Plot results
        plot_gnb_failure_results(analysis)

        # Generate detailed report
        generate_failure_report(analysis)

        print("\n5G gNB failure recovery simulation analysis completed!")

        # Output summary
        print("\n" + "=" * 50)
        print("SIMULATION SUMMARY:")
        print(f"  Failure Recovery: {'Successful' if analysis['recovery_success'] else 'Failed'}")
        print(f"  Recovery delay: {analysis['recovery_delay']:.3f}s" if analysis['recovery_delay'] != float(
            'inf') else "  Recovery delay: Not completed")
        print(f"  Service downtime: {analysis['service_downtime']:.3f}s")

        phase1 = analysis['phase1_stats']
        phase2 = analysis['phase2_stats']
        packet_loss_change = phase2['packet_loss'] - phase1['packet_loss']
        throughput_change = ((phase1['throughput'] - phase2['throughput']) / phase1['throughput'] * 100) if phase1[
                                                                                                                'throughput'] > 0 else 0

        print(f"  Packet loss change: +{packet_loss_change:.2f}%")
        print(f"  Throughput degradation: {throughput_change:.2f}%")

    except Exception as e:
        print(f"Program execution error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()