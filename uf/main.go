package main

import "fmt"

// UnionFind implementation
type UnionFind struct {
	parent map[int64]int64
	rank   map[int64]int
}

func NewUnionFind() *UnionFind {
	return &UnionFind{
		parent: make(map[int64]int64),
		rank:   make(map[int64]int),
	}
}

func (uf *UnionFind) MakeSet(x int64) {
	if _, exists := uf.parent[x]; !exists {
		uf.parent[x] = x
		uf.rank[x] = 0
	}
}

func (uf *UnionFind) Find(x int64) int64 {
	if uf.parent[x] != x {
		uf.parent[x] = uf.Find(uf.parent[x])
	}
	return uf.parent[x]
}

func (uf *UnionFind) Union(x, y int64) {
	uf.MakeSet(x)
	uf.MakeSet(y)

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
	uf := NewUnionFind()

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
