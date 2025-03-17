import m5
from m5.objects import *
from caches import *

import argparse

# binary = 'tests/test-progs/hello/bin/x86/linux/hello'

# to handle command line arguments
parser = argparse.ArgumentParser(description='A simple system with 2-level cache.')
parser.add_argument("binary", default="tests/test-progs/hello/bin/x86/linux/hello", nargs="?", type=str,
                    help="Path to the binary to execute.")
parser.add_argument("--l1i_size",
                    help=f"L1 instruction cache size. Default: 16kB.")
parser.add_argument("--l1d_size",
                    help="L1 data cache size. Default: Default: 64kB.")
parser.add_argument("--l2_size",
                    help="L2 cache size. Default: 256kB.")

options = parser.parse_args()

system = System()

# clock and voltage domains
system.clk_domain = SrcClockDomain()
system.clk_domain.clock = '1GHz'
system.clk_domain.voltage_domain = VoltageDomain()  # default

# 512MB of "timing" memory
system.mem_mode = 'timing'
system.mem_ranges = [AddrRange('512MB')]

# single cycle cpu for X86
system.cpu = X86TimingSimpleCPU()

# memory bus
system.membus = SystemXBar()

# L1 cache
system.cpu.icache = L1ICache(options)
system.cpu.dcache = L1DCache(options)

# connect L1 cache
system.cpu.icache.connectCPU(system.cpu)
system.cpu.dcache.connectCPU(system.cpu)

# # cache without Lx cache
# system.cpu.icache_port = system.membus.cpu_side_ports
# system.cpu.dcache_port = system.membus.cpu_side_ports

# connect L2 cache
system.l2bus = L2XBar()
system.cpu.icache.connectBus(system.l2bus)
system.cpu.dcache.connectBus(system.l2bus)
system.l2cache = L2Cache(options)
system.l2cache.connectCPUSideBus(system.l2bus)
system.l2cache.connectMemSideBus(system.membus)

# especially for X86
system.cpu.createInterruptController()
system.cpu.interrupts[0].pio = system.membus.mem_side_ports
system.cpu.interrupts[0].int_requestor = system.membus.cpu_side_ports
system.cpu.interrupts[0].int_responder = system.membus.mem_side_ports

system.system_port = system.membus.cpu_side_ports   # connect I/O

# memory controller
system.mem_ctrl = MemCtrl()
system.mem_ctrl.dram = DDR3_1600_8x8()
system.mem_ctrl.dram.range = system.mem_ranges[0]
system.mem_ctrl.port = system.membus.mem_side_ports




############################################################

# test program

# for gem5 V21 and beyond
# system.workload = SEWorkload.init_compatible(binary)
system.workload = SEWorkload.init_compatible(options.binary)      # use the binary from the command line

# create process for the test program
process = Process()
process.cmd = [options.binary]
system.cpu.workload = process
system.cpu.createThreads()

root = Root(full_system = False, system = system)   # SE mode
m5.instantiate()

print("Beginning simulation!")
exit_event = m5.simulate()

print('Exiting @ tick {} because {}'
      .format(m5.curTick(), exit_event.getCause()))