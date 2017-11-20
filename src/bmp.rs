#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Bmp8x8 {
    bytes: [u8; 8],
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Bit(pub u8);

impl Bmp8x8 {
    #[inline]
    pub fn read(&self, bit: Bit) -> bool {
        let (bit, i) = bit.unpack();
        (self.bytes[i] & bit) != 0
    }

    #[inline]
    pub fn write(&mut self, bit: Bit, value: bool) {
        let (bit, i) = bit.unpack();
        if value {
            self.bytes[i] |= bit;
        } else {
            self.bytes[i] &= !bit;
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        self.bytes = [0; 8];
    }
}

impl Bit {
    #[inline]
    fn unpack(self) -> (u8, usize) {
        let i = self.0;
        let x = 1 << (i % 8);
        let y = (i / 8) as usize;
        (x, y)
    }
}

