class Log:
    log = False

    @staticmethod
    def format_time(env_now):
        hours = int(env_now // 3600)
        minutes = int((env_now % 3600) // 60)
        seconds = round(env_now % 60, 4)
        return f"{hours:02}:{minutes:02}:{seconds:02}"

    @staticmethod
    def print_with_current_time(env_now, message):
        if Log.log:
            print(f"{Log.format_time(env_now)} {message}")
        else:
            pass

    @staticmethod
    def print_error(message):
        # Print message in red color
        if Log.log:
            print(f"\033[91m{message}\033[00m")
        else:
            pass

    @staticmethod
    def print_warning(message):
        # Print message in yellow color
        if Log.log:
            print(f"\033[93m{message}\033[00m")
        else:
            pass

    @staticmethod
    def print_success(message):
        # Print message in green color
        if Log.log:
            print(f"\033[92m{message}\033[00m")
        else:
            pass

    @staticmethod
    def print_simulation_results(qnodeList):
        if Log.log:
            total_waiting_time = 0
            total_execution_time = 0
            total_wall_time = 0
            # Create a list of all completed tasks across all QNodes
            all_completed_tasks = []
            all_failed_tasks = []
            for qnode in qnodeList:
                all_completed_tasks.extend(qnode.completed_tasks)
                all_failed_tasks.extend(qnode.failed_tasks)

            # Sort the tasks based on their IDs
            sorted_tasks = sorted(all_completed_tasks, key=lambda x: x.id)
            sorted_failed_tasks = sorted(all_failed_tasks, key=lambda x: x.id)
            print("=================================")
            print("Simulation Results:")
            print(f"✨ {len(all_completed_tasks)} SUCCESSFUL TASKS ✨")
            print("=================================")
            print(
                " QTask ID | QNode | Arrival Time | Waiting Time | Start Time   | Execution Time  | Wall Time   | Finish Time "
            )
            print(
                "----------|-------|--------------|--------------|--------------|-----------------|-------------|-------------"
            )

            # Print the sorted tasks
            for qtask in sorted_tasks:
                wall_time = qtask.waiting_time + qtask.execution_time
                print(
                    f" {qtask.id:^8} | {qtask.qnode.id:^5} | {round(qtask.arrival_time, 4):^12.4f} | {round(qtask.waiting_time, 4):^12.4f} | {round(qtask.start_running_time, 4):^12.4f} |  {round(qtask.execution_time, 4):^14.4f} | {round(wall_time, 4):^11.4f} | {round(qtask.finish_time, 4):^11.4f} "
                )
                # Accumulate the waiting and execution times
                total_waiting_time += qtask.waiting_time
                total_execution_time += qtask.execution_time
                total_wall_time += wall_time

            print("=================================")
            print(f"❌ {len(all_failed_tasks)} FAILED TASKS ❌")
            if len(all_failed_tasks) > 0:
                print(
                    " QTask ID | QNode | Arrival Time | Error                                                         "
                )
                print(
                    "----------|-------|--------------|---------------------------------------------------------------"
                )
                for qtask in sorted_failed_tasks:
                    print(
                        f" {qtask.id:^8} | {qtask.qnode.id:^5} | {round(qtask.arrival_time, 4):^12.4f} | {qtask.error:^9} "
                    )

            print(f"Total Waiting Time: {round(total_waiting_time, 2)}")
            print(f"Total Execution Time: {round(total_execution_time, 2)}")
            print(f"Total Wall Time: {round(total_wall_time, 2)}")
            print("QNode Relative Utilizations based on Share of Work:")
            for qnode in qnodeList:
                relative_utilization = qnode.total_busy_time / total_execution_time
                print(f"- QNode {qnode.id}: {relative_utilization:.2%}")
        else:
            pass
