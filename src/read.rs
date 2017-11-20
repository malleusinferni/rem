use std::collections::HashMap;

use super::*;

pub struct Instr {
    pub name: String,
    pub args: Vec<Arg>,
}

pub enum Arg {
    R(Reg),
    Rel(Reg, Reg),
    Abs(Reg, u32),
    Int(i32),
    Str(String),
}

pub enum Stmt {
    Label(String),
    Op(Op),
    Io(Io),
}

pub fn from_string(input: &str) -> Result<Program> {
    let mut blocks = HashMap::new();
    let mut atoms = atom::Table::new();

    atoms.intern("bye");

    for line in input.lines() {
        blocks.insert(Label(0), Block {
            name: line.to_owned(),
            kind: Cc::Jump,
            body: vec![],
            tail: Io::BYE,
        });
    }

    Ok(Program { blocks, atoms })
}

pub fn to_string(input: &Program) -> Result<String> {
    let mut buf = String::new();

    let mut labels = HashMap::new();

    for (label, block) in input.blocks.iter() {
        labels.insert(label, block.name.clone());
    }

    let a = |atom| input.atoms.resolve(atom);

    let l = |label| labels.get(&label)
        .ok_or(Error::NoSuchLabel { label });

    for (_label, block) in input.blocks.iter() {
        buf.push_str(&format!("{}:\n", &block.name));

        buf.push_str(&match block.kind {
            Cc::Func { env_index, arg_count } => {
                format!("func {}({}, {}):\n",
                &block.name, env_index, arg_count)
            },

            _ => format!("label {}:\n", &block.name),
        });

        let mut lines = Vec::<String>::new();

        for op in block.body.iter() {
            lines.push(match op.clone() {
                _ => format!("TODO")
            });
        }

        lines.push(match block.tail.clone() {
            Io::JUMP(label) => {
                format!("jump {}", l(label)?)
            },

            Io::IF(bit, cons, alt) => {
                format!("if {} then {} else {}", bit, l(cons)?, l(alt)?)
            },

            Io::TRACE(reg, label) => {
                format!("trace {} then {}", reg, l(label)?)
            },

            Io::SAY(reg, label) => {
                format!("say {} then {}", reg, l(label)?)
            },

            Io::SEND(lhs, rhs, label) => {
                format!("send {}, {} then {}", lhs, rhs, l(label)?)
            },

            Io::ARM(id, env, next) => {
                format!("arm {}({}) then {}", a(id), env, l(next)?)
            },

            Io::SPAWN(start, args, ret, next) => {
                format!("let {} = spawn {}({}) then {}",
                        ret, l(start)?, args, l(next)?)
            },

            Io::PUTENV(reg, id, next) => {
                format!("let ENV[{}] = {} then {}", id, reg, l(next)?)
            },

            Io::RECUR(start, args) => {
                format!("recur {}({})", l(start)?, args)
            },

            Io::RETI(retry) => if retry {
                format!("reject")
            } else {
                format!("resume")
            },

            Io::HCF => format!("hcf"),

            Io::BYE => format!("bye"),
        });

        for line in lines.drain(..) {
            buf.push('\t');
            buf.push_str(&line);
            buf.push('\n');
        }
    }

    Ok(buf)
}

use std::fmt;

impl fmt::Display for Reg {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "r{:02X}", self.0)
    }
}

impl fmt::Display for Bit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "b{:02X}", self.0)
    }
}
