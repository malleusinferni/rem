extern crate failure;
extern crate ordermap;
extern crate rayon;

#[macro_use]
extern crate failure_derive;

pub mod atom;
pub mod bmp;
pub mod read;
pub mod eval;

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::ops::{Deref, DerefMut};

use ordermap::OrderMap;

use atom::*;
use bmp::*;

/// Convenience alias with a default error type.
pub type Result<T, E=Error> = std::result::Result<T, E>;

/// Evaluation error with no debug information attached.
#[derive(Clone, Debug, Fail)]
pub enum Error {
    #[fail(display="User error")]
    Explicit,

    #[fail(display="Stack underflow")]
    StackUnderflow,

    #[fail(display="Divide by zero")]
    DividedByZero,

    #[fail(display="Malformed pattern")]
    MalformedPattern,

    #[fail(display="Register was used before it was initialized")]
    Uninitialized(Reg),

    #[fail(display="Value {:?} does not have expected type {:?}", got, wanted)]
    TypeMismatch { wanted: TypeTag, got: Value },

    #[fail(display="List index {} out of bounds", index)]
    OutOfBounds { index: Int },

    #[fail(display="No such label {:?}", label)]
    NoSuchLabel { label: Label },

    #[fail(display="No such environment ID {}", env_index)]
    NoSuchEnv { env_index: usize },

    #[fail(display="Illegal IO operation {:?}", io)]
    IllegalIo { io: Io },

    #[fail(display="Label {:?} expects calling convention {:?}", label, cc)]
    CcMismatch { label: Label, cc: Cc },

    #[fail(display="Expected {} arguments, found {}", wanted, got)]
    ArgCountMismatch { wanted: usize, got: usize },
}

pub type Int = i32;
pub type Str = Arc<str>;
pub type List = Arc<[Value]>;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Int(Int),
    Atom(Atom),
    Pid(Pid),
    Str(Str),
    List(List),
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum TypeTag {
    Int,
    Atom,
    Pid,
    Str,
    List,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Pattern {
    Wildcard,
    List(Vec<Pattern>),
    Value(Value),
}

pub struct Trap {
    env: List,
    handlers: Vec<Handler>,
}

#[derive(Clone)]
pub struct Handler(Label, [Pattern; 2]);

pub struct InterruptVector {
    message: [Value; 2],
    handlers: Vec<(List, Label)>,
}

/// Identifies a register in the virtual CPU.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Reg(pub u8);

#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub struct Label(pub usize);

/// Represents a basic block.
pub struct Block {
    pub name: String,
    pub kind: Cc,
    pub body: Vec<Op>,
    pub tail: Io,
}

pub struct Program {
    pub blocks: HashMap<Label, Block>,
    pub atoms: atom::Table,
}

/// Represents a successfully initialized program.
pub struct RunningProgram {
    envs: Box<[List]>,
    program: Program,
}

/// Encodes the calling convention a block expects.
#[derive(Copy, Clone, Debug)]
pub enum Cc {
    Init,
    Jump,
    Trap,
    Func {
        arg_count: usize,
        env_index: usize,
    },
}

/// Encodes computational instructions that can be executed by a Cpu alone.
///
/// Instruction arguments are ordered "write first," eg. the MOV instruction
/// reads its second argument and writes into the first. Arguments are also
/// named according to their purpose, so they can be pattern matched in
/// essentially any order.
#[derive(Clone, Debug)]
pub enum Op {
    NOP,

    MOV { dst: Reg, src: Rhs, },
    ADD { dst: Reg, src: Rhs, },
    SUB { dst: Reg, src: Rhs, },
    DIV { dst: Reg, src: Rhs, },
    MUL { dst: Reg, src: Rhs, },

    EQL { flag: Bit, lhs: Reg, rhs: Rhs, },
    GT { flag: Bit, lhs: Reg, rhs: Rhs, },
    LT { flag: Bit, lhs: Reg, rhs: Rhs, },
    GTE { flag: Bit, lhs: Reg, rhs: Rhs, },
    LTE { flag: Bit, lhs: Reg, rhs: Rhs, },

    SET { dst: Bit, val: bool, },
    CPY { dst: Bit, src: Bit, },

    /// Append a single value to the value buffer.
    VPUSH { src: Rhs, },

    /// Append the contents of a list to the value buffer.
    VCAT { src: Rhs, },

    /// Capture the value buffer as a list and save it in a register.
    VFIX { dst: Reg, },

    // TODO: STRPUSH

    /// Append the contents of a string to the string buffer.
    STRCAT { src: Rhs, },

    /// Capture the string buffer as a string and save it in a register.
    STRFIX { dst: Reg, },

    /// Append a wildcard to the pattern buffer.
    PATWILD,

    /// Append a single value to the pattern buffer.
    PATVAL { src: Rhs, },

    /// Turn the last `len` items in the pattern buffer into a list pattern.
    PATFIX { len: u32, },

    /// Move the contents of the pattern buffer to the trap buffer.
    PATPUSH { start: Label, },
}

/// Encodes control flow and IO operations that require supervisor assistance.
///
/// Most variants specify a return address. Several also specify a register to
/// write a return value into.
#[derive(Clone, Debug)]
pub enum Io {
    JUMP(Label),
    TRACE(Reg, Label),
    IF(Bit, Label, Label),
    SAY(Reg, Label),
    ARM(Atom, Reg, Label),
    SEND(Reg, Reg, Label),
    SPAWN(Label, Reg, Reg, Label),
    RECUR(Label, Reg),
    RETI(bool),
    PUTENV(Reg, usize, Label),
    HCF,
    BYE,
}

/// Encodes a literal value or a location in memory to be read from.
#[derive(Clone, Debug)]
pub enum Rhs {
    Reg(Reg),
    Int(Int),
    Str(Str),
    Rel(Reg, Reg),
    Abs(Reg, Int),
}

/// Main public interface for running processes.
pub struct Supervisor {
    program: RunningProgram,

    queue: HashMap<Pid, Process>,
    outbox: Vec<(Pid, List, Pid)>,
    spawner: Spawner,
    kill_list: Vec<Pid>,
}

struct Spawner {
    next_pid: Pid,
    processes: Vec<(Pid, Process)>,
}

#[derive(Clone, Debug)]
enum Status {
    Ready(Label),
    Blocked(Io),
    Dead(Error),
}

struct Interrupt {
    return_addr: Label,
    cpu: Box<eval::Cpu>,
    ivec: InterruptVector,
}

struct Process {
    main_cpu: Box<eval::Cpu>,
    interrupt: Option<Interrupt>,
    status: Status,
    traps: OrderMap<Atom, Trap>,
    inbox: VecDeque<(List, Pid)>,
}

#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub struct Pid(u32, u32); // Hey, why do I have two fields here

impl Status {
    #[inline]
    fn catch<F>(f: F) -> Self where F: FnOnce() -> Result<Self> {
        f().unwrap_or_else(Status::Dead)
    }
}

impl Pattern {
    fn matches(lhs: &[Self], rhs: &[Value]) -> bool {
        if lhs.len() != rhs.len() {
            return false;
        }

        lhs.iter().zip(rhs.iter()).all(|(lhs, rhs)| {
            match lhs {
                &Pattern::Wildcard => true,

                &Pattern::Value(ref lhs) => lhs == rhs,

                &Pattern::List(ref lhs) => match rhs {
                    &Value::List(ref rhs) => Pattern::matches(lhs, rhs),
                    _ => false,
                }
            }
        })
    }
}

impl Program {
    #[inline]
    fn jump(&self, label: Label) -> Result<&Block> {
        self.blocks.get(&label).ok_or(Error::NoSuchLabel { label })
    }

    fn init(self) -> Result<RunningProgram> {
        let mut envs = vec![];

        let nil = List::from(vec![]);
        let mut cpu = eval::Cpu::default();
        cpu.init(nil.clone(), &nil)?;

        let mut label = Label(0);

        loop {
            match cpu.eval(self.jump(label)?)? {
                Io::PUTENV(reg, id, next) => {
                    if id != envs.len() {
                        println!("This should be an error");
                    }

                    let env = cpu.read(reg)?.as_list()?;
                    envs.push(env);
                    label = next;
                },

                Io::JUMP(next) => label = next,

                Io::ARM(_, _, _) => continue, // TODO

                Io::BYE => break,

                io => return Err(Error::IllegalIo { io }),
            }
        }

        Ok(RunningProgram {
            program: self,
            envs: envs.into(),
        })
    }
}

impl RunningProgram {
    #[inline]
    fn call(&self, label: Label, args: &[Value]) -> Result<&List> {
        let block = self.jump(label)?;

        match block.kind {
            Cc::Func { arg_count, env_index } => if arg_count == args.len() {
                let env = self.envs.get(env_index)
                    .ok_or(Error::NoSuchEnv { env_index })?;

                Ok(env)
            } else {
                Err(Error::ArgCountMismatch {
                    wanted: arg_count,
                    got: args.len(),
                })
            },

            cc => Err(Error::CcMismatch { label, cc }),
        }
    }
}

impl InterruptVector {
    fn build(message: [Value; 2], traps: &OrderMap<Atom, Trap>) -> Self {
        let mut handlers = vec![];

        for (_id, trap) in traps.iter() { // FIXME: .rev()?
            for Handler(label, pat) in trap.handlers.iter().cloned() {
                if Pattern::matches(&pat, &message) {
                    handlers.push((trap.env.clone(), label));
                }
            }
        }

        InterruptVector { handlers, message }
    }

    fn next(&mut self) -> Option<(List, Label)> {
        self.handlers.pop()
    }
}

impl Supervisor {
    pub fn from_program(program: Program) -> Result<Self> {
        let spawner = Spawner {
            next_pid: Pid(0, 0),
            processes: vec![],
        };

        let program = program.init()?;

        Ok(Supervisor {
            program,
            spawner,
            queue: HashMap::new(),
            outbox: vec![],
            kill_list: vec![],
        })
    }

    pub fn spawn(&mut self, name: &str, args: &[Value]) -> Result<Pid> {
        let start = self.program.blocks.iter()
            .find(|&(_, ref block)| &block.name == name)
            .map(|(&label, _)| label )
            .ok_or(Error::Explicit)?;

        let env = self.program.call(start, args)?;
        self.spawner.process(start, env, args.into())
    }

    pub fn run(&mut self) {
        self.run_ready();
        self.send_messages();
        self.resume();
        self.receive_messages();
        self.cull_dead();
    }

    fn run_ready(&mut self) {
        use rayon::iter::*;

        // Borrow fields simultaneously. The borrow checker gets confused
        // sometimes if we just try to access them through self.
        let Supervisor { ref mut queue, ref program, .. } = *self;

        queue.par_iter_mut().for_each(|(_pid, process)| {
            if let Status::Ready(label) = process.status.clone() {
                process.status = Status::catch(|| {
                    let block = program.jump(label)?;
                    let io = process.eval(block)?;
                    Ok(Status::Blocked(io))
                });
            }
        });
    }

    fn send_messages(&mut self) {
        let Supervisor { ref mut queue, ref mut outbox, .. } = *self;

        for (&src, process) in queue.iter_mut() {
            process.status = match process.status {
                Status::Blocked(Io::SEND(dst, msg, next)) => {
                    Status::catch(|| {
                        let (dst, msg) = process.read_message(dst, msg)?;
                        outbox.push((dst, msg, src));
                        Ok(Status::Ready(next))
                    })
                },

                _ => continue,
            }
        }

        for (dst, msg, src) in outbox.drain(..) {
            if let Some(dst) = queue.get_mut(&dst) {
                dst.inbox.push_back((msg, src));
            } else {
                // Message is discarded
            }
            // TODO: Foreign/native processes
        }
    }

    fn resume(&mut self) {
        let Supervisor {
            ref mut queue,
            ref mut spawner,
            ref program,
            ..
        } = *self;

        for (&pid, process) in queue.iter_mut() {
            let io = match process.status.clone() {
                Status::Blocked(io) => io,
                _ => continue,
            };

            process.status = match io {
                Io::JUMP(label) => Status::Ready(label),

                Io::IF(bit, then_label, else_label) => {
                    if process.read_bit(bit) {
                        Status::Ready(then_label)
                    } else {
                        Status::Ready(else_label)
                    }
                },

                Io::TRACE(reg, label) => Status::catch(|| {
                    let val = process.trace(reg)?;
                    println!("{}", val);
                    Ok(Status::Ready(label))
                }),

                Io::ARM(id, env, next) => Status::catch(|| {
                    let (env, handlers) = process.arm(env)?;
                    process.traps.insert(id, Trap { env, handlers });
                    Ok(Status::Ready(next))
                }),

                Io::HCF => Status::Dead(Error::Explicit),

                Io::RETI(retry) => Status::catch(|| {
                    let Interrupt {
                        mut cpu,
                        mut ivec,
                        return_addr
                    } = process.interrupt.take()
                        .ok_or(Error::IllegalIo { io })?;

                    if retry {
                        while let Some((env, label)) = ivec.next() {
                            if let Err(_) = cpu.init(env, &ivec.message) {
                                continue;
                            }

                            process.interrupt = Some(Interrupt {
                                cpu,
                                ivec,
                                return_addr,
                            });

                            return Ok(Status::Ready(label));
                        }
                    }

                    Ok(Status::Ready(return_addr))
                }),

                Io::SPAWN(start, args, ret, next) => Status::catch(|| {
                    let args = process.read(args)?.as_list()?;
                    let env = program.call(start, &args)?;
                    let pid = spawner.process(start, &env, args)?;
                    process.write(ret, pid)?;
                    Ok(Status::Ready(next))
                }),

                Io::RECUR(start, args) => Status::catch(|| {
                    let args = process.read(args)?.as_list()?;
                    let env = program.call(start, &args)?;

                    if let Some(interrupt) = process.interrupt.take() {
                        // TODO: CPU pooling
                        let _ = interrupt.cpu;
                    }

                    // TODO: Global traps
                    process.traps.clear();

                    process.main_cpu.init(env.clone(), &args)?;

                    Ok(Status::Ready(start))
                }),

                // TODO: Other IO operations

                other => {
                    println!("{:?}: Unimplemented {:?}", pid, other);
                    continue;
                },
            };
        }

        queue.extend(spawner.drain());
    }

    fn receive_messages(&mut self) {
        let Supervisor { ref mut queue, ref mut spawner, .. } = *self;

        for (&_pid, process) in queue.iter_mut() {
            if process.interrupt.is_some() {
                return;
            }

            // Messages will only be delivered to unblocked processes
            // Call resume() before calling receive_messages()
            let return_addr = match process.status {
                Status::Ready(label) => label,
                _ => return,
            };

            let (body, src) = match process.inbox.pop_front() {
                Some(message) => message,
                None => return,
            };

            let message = [body.into(), src.into()];

            let mut ivec = {
                InterruptVector::build(message, &process.traps)
            };

            process.status = Status::catch(|| {
                while let Some((env, label)) = ivec.next() {
                    let cpu = match spawner.cpu(&env, &ivec.message) {
                        Ok(cpu) => cpu,
                        Err(_) => continue, // Don't crash the receiver
                    };

                    process.interrupt = Some(Interrupt {
                        cpu,
                        ivec,
                        return_addr,
                    });

                    return Ok(Status::Ready(label));
                }

                // Discard unhandled message and resume execution
                Ok(Status::Ready(return_addr))
            });
        }
    }

    fn cull_dead(&mut self) {
        let Supervisor {
            ref mut queue,
            ref mut spawner,
            ref mut kill_list,
            ..
        } = *self;

        for (&pid, process) in queue.iter_mut() {
            match process.status {
                Status::Dead(_) => kill_list.push(pid),
                Status::Ready(_) => continue,
                Status::Blocked(ref io) => {
                    println!("{:?} unexpectedly blocked on {:?}", pid, io);
                },
            }
        }

        for pid in kill_list.drain(..) {
            spawner.kill(queue.remove(&pid).unwrap());
        }
    }
}

impl Spawner {
    #[inline]
    fn cpu(&mut self, env: &List, args: &[Value]) -> Result<Box<eval::Cpu>> {
        // TODO: CPU pooling
        let mut cpu = eval::Cpu::default();
        cpu.init(env.clone(), args)?;
        Ok(cpu.into())
    }

    #[inline]
    fn process(&mut self, start: Label, env: &List, args: List) -> Result<Pid> {
        let pid = self.next_pid;
        self.next_pid.0 += 1; // TODO: Generation tagging, maybe

        let process = Process {
            main_cpu: self.cpu(env, &args)?.into(),
            interrupt: None,
            inbox: vec![].into(),
            traps: OrderMap::new(),
            status: Status::Ready(start),
        };

        self.processes.push((pid, process));

        Ok(pid)
    }

    #[inline]
    fn kill(&mut self, process: Process) {
        // TODO: CPU pooling
        let _ = process;
    }

    #[inline]
    fn drain(&mut self) -> std::vec::Drain<(Pid, Process)> {
        self.processes.drain(..)
    }
}

impl Deref for RunningProgram {
    type Target = Program;

    fn deref(&self) -> &Program {
        &self.program
    }
}

impl Deref for Process {
    type Target = eval::Cpu;

    fn deref(&self) -> &eval::Cpu {
        if let Some(interrupt) = self.interrupt.as_ref() {
            &interrupt.cpu
        } else {
            &self.main_cpu
        }
    }
}

impl DerefMut for Process {
    fn deref_mut(&mut self) -> &mut eval::Cpu {
        if let Some(interrupt) = self.interrupt.as_mut() {
            &mut interrupt.cpu
        } else {
            &mut self.main_cpu
        }
    }
}

impl Default for Value {
    fn default() -> Self {
        Value::List(Arc::new([]))
    }
}

impl<'a> From<&'a str> for Value {
    fn from(s: &'a str) -> Self {
        Value::Str(s.into())
    }
}

impl<'a, T: Clone + Into<Value>> From<&'a [T]> for Value {
    fn from(s: &'a [T]) -> Self {
        Value::List({
            s.iter().cloned().map(Into::into).collect::<Vec<_>>().into()
        })
    }
}

impl<T: Into<Value>> From<Vec<T>> for Value {
    fn from(v: Vec<T>) -> Self {
        Value::List({
            v.into_iter().map(Into::into).collect::<Vec<_>>().into()
        })
    }
}

impl From<Str> for Value {
    fn from(s: Str) -> Self {
        Value::Str(s)
    }
}

impl From<List> for Value {
    fn from(l: List) -> Self {
        Value::List(l)
    }
}

impl From<Int> for Value {
    fn from(i: Int) -> Self {
        Value::Int(i)
    }
}

impl From<Pid> for Value {
    fn from(pid: Pid) -> Self {
        Value::Pid(pid)
    }
}

impl From<Reg> for Rhs {
    fn from(r: Reg) -> Self {
        Rhs::Reg(r)
    }
}

#[test]
fn hello_world() {
    let atoms = atom::Table::new();

    let blocks = vec!{
        Block {
            name: "init".into(),
            kind: Cc::Init,
            body: vec!{},
            tail: Io::PUTENV(Reg(0), 0, Label(1)),
        },

        Block {
            name: "exit".into(),
            kind: Cc::Jump,
            body: vec![],
            tail: Io::BYE,
        },

        Block {
            name: "hello".into(),
            kind: Cc::Func {
                env_index: 0,
                arg_count: 0,
            },
            body: vec!{
                Op::MOV { dst: Reg(2), src: Rhs::Str("Hello, world".into()), },
            },
            tail: Io::TRACE(Reg(2), Label(1)),
        },
    }.into_iter().enumerate().map(|(i, block)| {
        (Label(i), block)
    }).collect();

    let mut supervisor = Supervisor::from_program(Program { blocks, atoms })
        .expect("Whoops");

    supervisor.spawn("hello", &[]).unwrap();
    supervisor.run();
}

