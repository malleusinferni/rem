use super::*;
use bmp::*;
use atom::*;

#[derive(Default)]
pub struct Cpu {
    flags: Bmp8x8,
    val_buf: Vec<Value>,
    str_buf: String,
    pat_buf: Vec<Pattern>,
    trap_buf: Vec<Handler>,
    registers: Vec<Value>,
}

impl Cpu {
    pub fn init(&mut self, env: List, args: &[Value]) -> Result<()> {
        self.flags.clear();
        self.val_buf.clear();
        self.str_buf.clear();
        self.pat_buf.clear();
        self.trap_buf.clear();
        self.registers.clear();

        self.write(Reg(0), env)?;
        self.write(Reg(1), args)?;
        Ok(())
    }

    pub fn eval(&mut self, block: &Block) -> Result<Io> {
        for op in block.body.iter().cloned() {
            self.step(op)?;
        }

        Ok(block.tail.clone())
    }

    pub fn read_message(&self, dst: Reg, msg: Reg) -> Result<(Pid, List)> {
        let dst = self.read(dst)?.as_pid()?;
        let msg = self.read(msg)?.as_list()?;
        Ok((dst, msg))
    }

    pub fn read_bit(&self, bit: Bit) -> bool {
        self.flags.read(bit)
    }

    pub fn trace(&self, rhs: Reg) -> Result<String> {
        let rhs = self.read(rhs)?;
        Ok(format!("{:?}", rhs))
    }

    pub fn arm(&mut self, env: Reg) -> Result<(List, Vec<Handler>)> {
        let env = self.read(env)?.as_list()?;
        Ok((env, self.trap_buf.drain(..).collect()))
    }

    #[inline]
    pub fn step(&mut self, op: Op) -> Result<()> {
        match op {
            Op::NOP => (),

            Op::MOV { dst, src } => {
                let src = self.read(src)?;
                self.write(dst, src)?;
            },

            Op::ADD { dst, src } => {
                let src = self.read(src)?.as_int()?;
                self.with_lhs(dst, |lhs| {
                    Ok(lhs.as_int()?.wrapping_add(src))
                })?;
            },

            Op::SUB { dst, src } => {
                let src = self.read(src)?.as_int()?;
                self.with_lhs(dst, |lhs| {
                    Ok(lhs.as_int()?.wrapping_sub(src))
                })?;
            },

            Op::DIV { dst, src } => {
                let src = self.read(src)?.as_int()?;
                if src == 0 { return Err(Error::DividedByZero); }
                self.with_lhs(dst, |lhs| {
                    Ok(lhs.as_int()? / src)
                })?;
            },

            Op::MUL { dst, src } => {
                let src = self.read(src)?.as_int()?;
                self.with_lhs(dst, |lhs| {
                    Ok(lhs.as_int()?.wrapping_mul(src))
                })?;
            },

            Op::EQL { flag, lhs, rhs } => {
                let rhs = self.read(rhs)?;
                let lhs = self.read(lhs)?;
                self.flags.write(flag, lhs == rhs);
            },

            Op::GTE { flag, lhs, rhs } => {
                let rhs = self.read(rhs)?.as_int()?;
                let lhs = self.read(lhs)?.as_int()?;
                self.flags.write(flag, lhs >= rhs);
            },

            Op::LTE { flag, lhs, rhs } => {
                let rhs = self.read(rhs)?.as_int()?;
                let lhs = self.read(lhs)?.as_int()?;
                self.flags.write(flag, lhs <= rhs);
            },

            Op::GT { flag, lhs, rhs } => {
                let rhs = self.read(rhs)?.as_int()?;
                let lhs = self.read(lhs)?.as_int()?;
                self.flags.write(flag, lhs > rhs);
            },

            Op::LT { flag, lhs, rhs } => {
                let rhs = self.read(rhs)?.as_int()?;
                let lhs = self.read(lhs)?.as_int()?;
                self.flags.write(flag, lhs < rhs);
            },

            Op::CPY { dst, src } => {
                let b = self.flags.read(src);
                self.flags.write(dst, b);
            },

            Op::SET { dst, val } => {
                self.flags.write(dst, val);
            },

            Op::STRCAT { src } => {
                let string = self.read(src)?.as_str()?;
                self.str_buf.push_str(&string);
            },

            Op::STRFIX { dst } => {
                let src = Value::from(self.str_buf.as_str());
                self.str_buf.clear();
                self.write(dst, src)?;
            },

            Op::VPUSH { src } => {
                let elem = self.read(src)?;
                self.val_buf.push(elem);
            },

            Op::VCAT { src } => {
                let list = self.read(src)?.as_list()?;
                self.val_buf.extend(list.iter().cloned());
            },

            Op::VFIX { dst } => {
                let list = List::from(self.val_buf.as_slice());
                self.val_buf.clear();
                self.write(dst, list)?;
            },

            Op::PATWILD => {
                self.pat_buf.push(Pattern::Wildcard);
            },

            Op::PATVAL { src } => {
                let pat = self.read(src)?;
                self.pat_buf.push(Pattern::Value(pat));
            },

            Op::PATFIX { len } => {
                let start = self.pat_buf.len()
                    .checked_sub(len as usize)
                    .ok_or(Error::MalformedPattern)?;

                let pat = self.pat_buf.drain(start ..).collect();
                self.pat_buf.push(Pattern::List(pat));
            },

            Op::PATPUSH { start } => {
                let sender = self.pat_buf.pop()
                    .ok_or(Error::MalformedPattern)?;
                let body = self.pat_buf.pop()
                    .ok_or(Error::MalformedPattern)?;

                if self.pat_buf.len() > 0 {
                    return Err(Error::MalformedPattern);
                }

                self.trap_buf.push(Handler(start, [body, sender]));
            },
        }

        Ok(())
    }

    #[inline]
    pub fn read<R: Into<Rhs>>(&self, value: R) -> Result<Value> {
        match value.into() {
            Rhs::Reg(r) => {
                let i = r.0 as usize;

                if i < self.registers.len() {
                    Ok(self.registers[i].clone())
                } else {
                    Err(Error::Uninitialized(r))
                }
            },

            Rhs::Rel(ptr, idx) => {
                let list = self.read(ptr)?;
                let idx = self.read(idx)?.as_int()?;
                list.index(idx)
            },

            Rhs::Abs(ptr, idx) => {
                let list = self.read(ptr)?;
                list.index(idx)
            },

            Rhs::Int(i) => Ok(i.into()),
            Rhs::Str(s) => Ok(s.into()),
        }
    }

    #[inline]
    pub fn write<V>(&mut self, reg: Reg, value: V) -> Result<()>
        where V: Into<Value>
    {
        let i = reg.0 as usize;
        let value = value.into();

        if i == self.registers.len() {
            self.registers.push(value);
        } else if i < self.registers.len() {
            self.registers[i] = value;
        } else {
            return Err(Error::Uninitialized(reg));
        }

        Ok(())
    }

    #[inline]
    fn with_lhs<F, V>(&mut self, reg: Reg, op: F) -> Result<()>
        where F: FnOnce(Value) -> Result<V>, V: Into<Value>
    {
        let lhs = self.read(reg)?;
        self.write(reg, op(lhs)?)
    }
}

impl Value {
    pub fn index(self, offset: Int) -> Result<Value> {
        let list = self.as_list()?;
        if 0 <= offset && offset < list.len() as Int {
            Ok(list[offset as usize].clone())
        } else {
            Err(Error::OutOfBounds { index: offset })
        }
    }

    pub fn as_int(self) -> Result<Int> {
        match self {
            Value::Int(i) => Ok(i),
            other => Err(Error::TypeMismatch {
                wanted: TypeTag::Int,
                got: other,
            }),
        }
    }

    pub fn as_str(self) -> Result<Str> {
        match self {
            Value::Str(s) => Ok(s),
            other => Err(Error::TypeMismatch {
                wanted: TypeTag::Str,
                got: other,
            }),
        }
    }

    pub fn as_list(self) -> Result<List> {
        match self {
            Value::List(v) => Ok(v),
            other => Err(Error::TypeMismatch {
                wanted: TypeTag::List,
                got: other,
            }),
        }
    }

    pub fn as_atom(self) -> Result<Atom> {
        match self {
            Value::Atom(a) => Ok(a),
            other => Err(Error::TypeMismatch {
                wanted: TypeTag::Atom,
                got: other,
            }),
        }
    }

    pub fn as_bool(self) -> Result<bool> {
        self.as_int().map(|i| i != 0)
    }

    pub fn as_pid(self) -> Result<Pid> {
        match self {
            Value::Pid(pid) => Ok(pid),
            other => Err(Error::TypeMismatch {
                wanted: TypeTag::Pid,
                got: other,
            }),
        }
    }
}

#[cfg(test)]
impl Cpu {
    fn run(program: Vec<Op>) -> Self {
        let mut cpu = Self::default();

        for op in program {
            cpu.step(op).unwrap();
        }

        cpu
    }

    fn expect<V: Into<Value>>(self, reg: Reg, value: V) -> Self {
        let result = self.read(reg).unwrap();
        assert_eq!(result, value.into());
        self
    }
}

#[test]
fn simple_program() {
    let program = vec![
        Op::MOV { dst: Reg(0), src: Rhs::Int(2), },
        Op::ADD { dst: Reg(0), src: Rhs::Reg(Reg(0)), },
    ];

    Cpu::run(program).expect(Reg(0), 4);
}

#[test]
fn list_manip() {
    Cpu::run(vec!{
        Op::VPUSH { src: Rhs::Str("Hello".into()) },
        Op::VPUSH { src: Rhs::Str("world".into()) },
        Op::VFIX { dst: Reg(0) },
    });
}

#[test]
fn pat_manip() {
    let pattern = Cpu::run(vec!{
        Op::PATWILD,
        Op::PATWILD,
        Op::PATWILD,
        Op::PATFIX { len: 2 },
    }).pat_buf;

    let message: &[Value] = &[ 1.into(), vec![1i32, 2].into(), ];

    assert_eq!(pattern.len(), 2);
    assert!(Pattern::matches(&pattern, &message));
}

