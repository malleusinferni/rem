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

    let mut lines = Vec::<String>::new();

    for (label, block) in input.blocks.iter() {
        labels.insert(label, block.name.clone());
    }

    for (_label, block) in input.blocks.iter() {
        buf.push_str(&format!("{}:\n", &block.name));

        for op in block.body.iter() {
            lines.push(match op.clone() {
                _ => format!("TODO")
            });
        }

        lines.push(match block.tail.clone() {
            Io::JUMP(label) => {
                let label = labels.get(&label).unwrap();
                format!("jump {}", label)
            },

            Io::IF(bit, cons, alt) => {
                let cons = labels.get(&cons).unwrap();
                let alt = labels.get(&alt).unwrap();
                format!("ifelse {}, {}, {}", bit, cons, alt)
            },

            Io::TRACE(reg, label) => {
                let label = labels.get(&label).unwrap();
                format!("trace {}, {}", reg, label)
            },

            Io::SAY(reg, label) => {
                let label = labels.get(&label).unwrap();
                format!("say {}, {}", reg, label)
            },

            Io::SEND(lhs, rhs, label) => {
                let label = labels.get(&label).unwrap();
                format!("send {}, {}, {}", lhs, rhs, label)
            },

            Io::ARM(id, env, next) => {
                let id = input.atoms.resolve(id);
                let next = labels.get(&next).unwrap();
                format!("arm {}, {}, {}", id, env, next)
            },

            Io::SPAWN(label, env, ret, next) => {
                let label = labels.get(&label).unwrap();
                let next = labels.get(&next).unwrap();
                format!("spawn {}, {}, {}, {}", label, env, ret, next)
            },

            Io::PUTENV(reg, id, next) => {
                let next = labels.get(&next).unwrap();
                format!("env[{}] = {} ~> {}", id, reg, next)
            },

            Io::RECUR(start, args) => {
                let start = labels.get(&start).unwrap();
                format!("recur {}, {}", start, args)
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
