package main

import "fmt"

// UnionFind implementation
type UnionFind struct {
	parent []uint32
	rank   []uint32
}

const unset uint32 = ^uint32(0)

func NewUnionFind(maxElement uint32) *UnionFind {
	if maxElement == unset {
		panic("maxElement must be less than unset")
	}

	n := int(maxElement) + 1
	parent := make([]uint32, n)
	rank := make([]uint32, n)
	for i := 0; i < n; i++ {
		parent[i] = uint32(i)
	}

	return &UnionFind{
		parent: parent,
		rank:   rank,
	}
}

func (uf *UnionFind) validateElement(x uint32) {
	if x == unset {
		panic("element value unset is reserved")
	}
	if int(x) >= len(uf.parent) {
		panic("element out of range")
	}
}

func (uf *UnionFind) Find(x uint32) uint32 {
	uf.validateElement(x)
	if uf.parent[x] != x {
		uf.parent[x] = uf.Find(uf.parent[x])
	}
	return uf.parent[x]
}

func (uf *UnionFind) Union(x, y uint32) {
	uf.validateElement(x)
	uf.validateElement(y)

	rootX := uf.Find(x)
	rootY := uf.Find(y)

	if rootX == rootY {
		return
	}

	if uf.rank[rootX] < uf.rank[rootY] {
		uf.parent[rootX] = rootY
	} else if uf.rank[rootX] > uf.rank[rootY] {
		uf.parent[rootY] = rootX
	} else {
		uf.parent[rootY] = rootX
		uf.rank[rootX]++
	}
}

func (uf *UnionFind) Size() int {
	return len(uf.parent)
}

func main() {
	uf := NewUnionFind(10)

	// Объединяем 1 и 2 -> группа {1,2}
	uf.Union(1, 2)

	// Объединяем 3 и 4 -> группа {3,4}
	uf.Union(3, 4)

	// Проверим корни до общего объединения
	fmt.Println("Find(1):", uf.Find(1))
	fmt.Println("Find(2):", uf.Find(2))
	fmt.Println("Find(3):", uf.Find(3))
	fmt.Println("Find(4):", uf.Find(4))

	// Объединяем две группы через 2 и 3 -> теперь {1,2,3,4}
	uf.Union(2, 3)

	fmt.Println("После Union(2,3):")
	fmt.Println("Find(1):", uf.Find(1))
	fmt.Println("Find(2):", uf.Find(2))
	fmt.Println("Find(3):", uf.Find(3))
	fmt.Println("Find(4):", uf.Find(4))

	// Сколько всего элементов видит UnionFind
	fmt.Println("Size:", uf.Size()) // 4
}
