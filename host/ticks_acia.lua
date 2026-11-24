-- z88dk-ticks port hook, loaded by host/z80nn.py ("emu" target) with -lua.
--
-- ticks already emulates the board's MC6850 ACIA (ports 0x80/0x81) on its
-- stdin/stdout, but never flushes stdout: through a pipe the replies of
-- build/board.rom stay in the buffer and the host waits forever.
-- This forwards every OUT to the built-in ACIA and flushes after each data byte.
-- Ports that are not the ACIA return -1 and fall through to ticks' other handlers.

function io_out(port, value)
    local handled = ticks.acia_out(port, value)
    if port & 0xff == 0x81 then
        io.stdout:flush()
    end
    return handled
end
