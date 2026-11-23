/* Custom Logic for DeZog zsim: the ports of firmware/z80_draw_main.c.
 *
 *   IN  0x10  next byte of the UI's queued messages, 0 when nothing is queued
 *   OUT 0x11  one byte of a reply
 *   OUT 0x12  end of reply, value = type ('A', 'R', 'G', 'M')
 *
 * Messages from the UI:   { command: 'send', bytes: [...] }   (one command each)
 * Messages to the UI:     { command: 'ready' }
 *                         { command: 'started', type, tstates }   Z80 read a command byte
 *                         { command: 'reply', type, bytes, tstates, cycles? }
 *
 * Only the low 8 bits of the port address are decoded (IN A,(n) puts A on the
 * high byte).
 */
const PORT_IN = 0x10, PORT_OUT = 0x11, PORT_END = 0x12;

let messages = [];      // queued UI messages, each an array of bytes
let current = null;     // the message the Z80 is reading
let pos = 0;
let reply = [];         // bytes OUTed since the last end-of-reply
let inferStart;         // t-states when the Z80 read an 'I' command
let addresses;          // last 'A' reply, re-sent when the UI (re)loads

API.uiReady = () => {
    API.sendToCustomUi({ command: 'ready', queued: messages.length });
    if (addresses) API.sendToCustomUi(addresses);
};

API.receivedFromCustomUi = (msg) => {
    if (msg.command === 'send' && msg.bytes && msg.bytes.length > 0)
        messages.push(msg.bytes.map(b => b & 0xFF));
};

API.readPort = (port) => {
    if ((port & 0xFF) !== PORT_IN) return undefined;
    if (current === null) {
        if (messages.length === 0) return 0;
        current = messages.shift();
        pos = 0;
        const type = String.fromCharCode(current[0]);
        if (type === 'I') inferStart = API.tstates;
        API.sendToCustomUi({ command: 'started', type, tstates: API.tstates });
    }
    const b = current[pos++];
    if (pos >= current.length) current = null;
    return b;
};

API.writePort = (port, value) => {
    const p = port & 0xFF;
    if (p === PORT_OUT) {
        reply.push(value & 0xFF);
    } else if (p === PORT_END) {
        const msg = {
            command: 'reply',
            type: String.fromCharCode(value & 0xFF),
            bytes: reply,
            tstates: API.tstates
        };
        if (msg.type === 'R' && inferStart !== undefined) {
            msg.cycles = API.tstates - inferStart;
            inferStart = undefined;
        }
        if (msg.type === 'A') addresses = msg;
        reply = [];
        API.sendToCustomUi(msg);
    }
};
