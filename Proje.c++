/**
* Projem
* Classes and Inheritance:
*The code comprises several classes: Imaginary, Complex, Matrix, Vector, SquareMatrix, IdentityMatrix, TriangleMatrix, and LinearAlgebraObject.
*Inheritance is utilized, where SquareMatrix, IdentityMatrix, and TriangleMatrix inherit from the Matrix class, and LinearAlgebraObject is an abstract base class.
*Functionality:
*Imaginary and Complex classes handle complex numbers, supporting arithmetic operations and manipulation of real and imaginary parts.
*Matrix and Vector classes manage matrices and vectors, respectively, allowing operations such as addition, subtraction, multiplication, transpose, determinant calculation, and checking equality.
*SquareMatrix, IdentityMatrix, and TriangleMatrix classes offer specialized functionalities for square matrices, identity matrices, and triangular matrices, respectively.
*Operator overloading is implemented extensively for various mathematical operations on these algebraic structures.
* 2 Proje
* 02\01\2024
* Rüya alabed
* roua.alabed@stu.fsm.edu.tr
*/


#include <iostream>
#include <vector>

class Imaginary
{
private:
    int im; // Imaginary part of a complex number

public:
    Imaginary(int value) : im(value) {}

    int getImaginary() const  // Getter for the imaginary part
    {
        return im;
    }

    void setImaginary(int value)  // Setter for the imaginary part
    { 
        im = value;
    }
};

class Complex
{
private:
    int re;         // Real part of a complex number
    Imaginary im;   // Imaginary part of a complex number

public:
    Complex() : re(0), im(0) {} // Default constructor

    Complex(int real, int imag) : re(real), im(imag) {} // Parameterized constructor
    
    // Getters and setters for real and imaginary parts
    int getReal() const
    {
        return re;
    }

    void setReal(int value)
    {
        re = value;
    }

    int getImaginary() const
    {
        return im.getImaginary();
    }

    void setImaginary(int value)
    {
        im.setImaginary(value);
    }
    // Overloaded arithmetic operators for complex numbers
    Complex operator+(const Complex &other) const
    {   
        // Addition of two complex numbers
        int newReal = this->re + other.re;
        int newImag = this->im.getImaginary() + other.im.getImaginary();
        return Complex(newReal, newImag);
    }

    Complex operator-(const Complex &other) const
    {
        int newReal = this->re - other.re;
        int newImag = this->im.getImaginary() - other.im.getImaginary();
        return Complex(newReal, newImag);
    }

    Complex operator*(const Complex &other) const
    {
        int newReal = (this->re * other.re) - (this->im.getImaginary() * other.im.getImaginary());
        int newImag = (this->re * other.im.getImaginary()) + (this->im.getImaginary() * other.re);
        return Complex(newReal, newImag);
    }

    Complex operator*(int value) const
    {
        return Complex(this->re * value, this->im.getImaginary() * value);
    }

    Complex operator/(const Complex &other)
    {
        int divisor = (other.re * other.re) + (other.im.getImaginary() * other.im.getImaginary());
        if (divisor == 0)
        {
            throw std::runtime_error("Division by zero is not allowed");
        }
        int newReal = ((this->re * other.re) + (this->im.getImaginary() * other.im.getImaginary())) / divisor;
        int newImag = ((this->im.getImaginary() * other.re) - (this->re * other.im.getImaginary())) / divisor;
        return Complex(newReal, newImag);
    }

    Complex &operator+=(const Complex &other)
    {
        this->re += other.re;
        this->im.setImaginary(this->im.getImaginary() + other.im.getImaginary());
        return *this;
    }

    Complex &operator-=(const Complex &other)
    {
        this->re -= other.re;
        this->im.setImaginary(this->im.getImaginary() - other.im.getImaginary());
        return *this;
    }

    Complex &operator*=(const Complex &other)
    {
        int newReal = (this->re * other.re) - (this->im.getImaginary() * other.im.getImaginary());
        int newImag = (this->re * other.im.getImaginary()) + (this->im.getImaginary() * other.re);
        this->re = newReal;
        this->im.setImaginary(newImag);
        return *this;
    }

    Complex &operator/=(const Complex &other)
    {
        int divisor = (other.re * other.re) + (other.im.getImaginary() * other.im.getImaginary());
        int newReal = ((this->re * other.re) + (this->im.getImaginary() * other.im.getImaginary())) / divisor;
        int newImag = ((this->im.getImaginary() * other.re) - (this->re * other.im.getImaginary())) / divisor;
        this->re = newReal;
        this->im.setImaginary(newImag);
        return *this;
    }

    bool operator==(const Complex &other) const
    {
        return (this->re == other.re) && (this->im.getImaginary() == other.im.getImaginary());
    }

    bool operator!=(const Complex &other) const
    {
        return !(*this == other);
    }
};
class Matrix
{
private:
    size_t row, col;
    std::vector<std::vector<Complex>> elements;

public:
    Matrix(size_t r, size_t c) : row(r), col(c), elements(r, std::vector<Complex>(c)) {}

    // Getter for number of rows
    size_t getRows() const
    {
        return row;
    }

    // Getter for number of columns
    size_t getCols() const
    {
        return col;
    }

    // Method to access a specific element in the matrix
    Complex &at(size_t r, size_t c)
    {
        return elements[r][c];
    }

    // Method to print the matrix
    void print() const
    {
        for (size_t i = 0; i < row; ++i)
        {
            for (size_t j = 0; j < col; ++j)
            {
                std::cout << elements[i][j].getReal() << "+" << elements[i][j].getImaginary() << "i\t";
            }
            std::cout << std::endl;
        }
    }
    Matrix operator+(const Matrix &otherMatrix) const
    {
        if (row != otherMatrix.row || col != otherMatrix.col)
        {
            throw std::runtime_error("Matrix sizes are not compatible for addition");
        }

        Matrix result(row, col);
        for (size_t i = 0; i < row; ++i)
        {
            for (size_t j = 0; j < col; ++j)
            {
                result.elements[i][j] = elements[i][j] + otherMatrix.elements[i][j];
            }
        }
        return result;
    }

    Matrix operator-(const Matrix &otherMatrix) const
    {
        if (row != otherMatrix.row || col != otherMatrix.col)
        {
            throw std::runtime_error("Matrix sizes are not compatible for subtraction");
        }

        Matrix result(row, col);
        for (size_t i = 0; i < row; ++i)
        {
            for (size_t j = 0; j < col; ++j)
            {
                result.elements[i][j] = elements[i][j] - otherMatrix.elements[i][j];
            }
        }
        return result;
    }

    Matrix operator*(const Matrix &otherMatrix) const
    {
        if (col != otherMatrix.row)
        {
            throw std::runtime_error("Matrix sizes are not compatible for multiplication");
        }

        Matrix result(row, otherMatrix.col);
        for (size_t i = 0; i < row; ++i)
        {
            for (size_t j = 0; j < otherMatrix.col; ++j)
            {
                for (size_t k = 0; k < col; ++k)
                {
                    result.elements[i][j] += elements[i][k] * otherMatrix.elements[k][j];
                }
            }
        }
        return result;
    }

    Matrix &operator=(const Matrix &otherMatrix)
    {
        if (&otherMatrix != this)
        {
            row = otherMatrix.row;
            col = otherMatrix.col;
            elements = otherMatrix.elements;
        }
        return *this;
    }

    Matrix &operator+=(const Matrix &otherMatrix)
    {
        if (row != otherMatrix.row || col != otherMatrix.col)
        {
            throw std::runtime_error("Matrix sizes are not compatible for compound addition");
        }

        for (size_t i = 0; i < row; ++i)
        {
            for (size_t j = 0; j < col; ++j)
            {
                elements[i][j] += otherMatrix.elements[i][j];
            }
        }
        return *this;
    }

    Matrix &operator-=(const Matrix &otherMatrix)
    {
        if (row != otherMatrix.row || col != otherMatrix.col)
        {
            throw std::runtime_error("Matrix sizes are not compatible for compound subtraction");
        }

        for (size_t i = 0; i < row; ++i)
        {
            for (size_t j = 0; j < col; ++j)
            {
                elements[i][j] -= otherMatrix.elements[i][j];
            }
        }
        return *this;
    }

    Matrix &operator*=(const Matrix &otherMatrix)
    {
        if (col != otherMatrix.row)
        {
            throw std::runtime_error("Matrix sizes are not compatible for compound multiplication");
        }

        // Matrix multiplication here.
        Matrix result = (*this) * otherMatrix;
        *this = result;
        return *this;
    }

    bool operator==(const Matrix &otherMatrix) const
    {
        if (row != otherMatrix.row || col != otherMatrix.col)
        {
            return false;
        }

        for (size_t i = 0; i < row; ++i)
        {
            for (size_t j = 0; j < col; ++j)
            {
                if (elements[i][j] != otherMatrix.elements[i][j])
                {
                    return false;
                }
            }
        }
        return true;
    }
    // Method to calculate the transpose of the matrix
    Matrix transpose() const
    {
        Matrix transposed(col, row);

        for (size_t i = 0; i < row; ++i)
        {
            for (size_t j = 0; j < col; ++j)
            {
                transposed.elements[j][i] = elements[i][j];
            }
        }

        return transposed;
    }

    const Complex &at(size_t r, size_t c) const
    {
        return elements[r][c];
    }

    // Method to calculate the determinant of the matrix
    Complex determinant() const
    {
        if (getRows() != getCols())
        {
            // Handle error - determinant calculation is only for square matrices
            throw std::runtime_error("Determinant can only be calculated for square matrices");
        }

        size_t matrixSize = getRows();

        if (matrixSize == 1)
        {
            // For a 1x1 matrix, the determinant is the single element itself
            return at(0, 0);
        }
        else if (matrixSize == 2)
        {
            // For a 2x2 matrix, use the simple formula ad - bc
            return (at(0, 0) * at(1, 1)) - (at(0, 1) * at(1, 0));
        }
        else
        {
            // For larger matrices, implement the determinant calculation using cofactor expansion
            Complex det(0, 0);
            int sign = 1;

            for (size_t i = 0; i < matrixSize; ++i)
            {
                // Calculate the cofactor matrix
                Matrix submatrix(matrixSize - 1, matrixSize - 1);
                for (size_t j = 1; j < matrixSize; ++j)
                {
                    size_t colIndex = 0;
                    for (size_t k = 0; k < matrixSize; ++k)
                    {
                        if (k == i)
                        {
                            continue;
                        }
                        submatrix.at(j - 1, colIndex) = at(j, k);
                        colIndex++;
                    }
                }

                // Calculate the determinant recursively for the submatrix
                Complex subDet = submatrix.determinant();

                // Add to the determinant using cofactor expansion
                det += at(0, i) * subDet * sign;
                sign = -sign;
            }

            return det;
        }
        return Complex(0, 0); // Default value; replace this with your actual calculation
    }
};

class Vector
{
private:
    size_t size;
    std::vector<Complex> elements;

public:
    Vector(size_t s) : size(s), elements(s) {}

    // Getter for size
    size_t getSize() const { return size; }

    // Method to access an element in the vector
    Complex &at(size_t index) { return elements[index]; }

    Vector operator+(const Vector &otherVector) const
    {
        if (size != otherVector.size)
        {
            throw std::runtime_error("Vector sizes are not compatible for addition");
        }

        Vector result(size);
        for (size_t i = 0; i < size; ++i)
        {
            result.elements[i] = elements[i] + otherVector.elements[i];
        }
        return result;
    }

    Vector operator-(const Vector &otherVector) const
    {
        if (size != otherVector.size)
        {
            throw std::runtime_error("Vector sizes are not compatible for subtraction");
        }

        Vector result(size);
        for (size_t i = 0; i < size; ++i)
        {
            result.elements[i] = elements[i] - otherVector.elements[i];
        }
        return result;
    }

    // Operator overloading for vector multiplication (dot product) or matrix multiplication
    Complex operator*(const Vector &otherVector) const
    {
        if (size != otherVector.size)
        {
            throw std::runtime_error("Vector sizes are not compatible for multiplication");
        }

        Complex result = elements[0] * otherVector.elements[0];
        for (size_t i = 1; i < size; ++i)
        {
            result += elements[i] * otherVector.elements[i];
        }
        return result;
    }

    Vector &operator=(const Vector &otherVector)
    {
        if (&otherVector != this)
        {
            size = otherVector.size;
            elements = otherVector.elements;
        }
        return *this;
    }

    Vector &operator+=(const Vector &otherVector)
    {
        if (size != otherVector.size)
        {
            throw std::runtime_error("Vector sizes are not compatible for compound addition");
        }

        for (size_t i = 0; i < size; ++i)
        {
            elements[i] += otherVector.elements[i];
        }
        return *this;
    }

    Vector &operator-=(const Vector &otherVector)
    {
        if (size != otherVector.size)
        {
            throw std::runtime_error("Vector sizes are not compatible for compound subtraction");
        }

        for (size_t i = 0; i < size; ++i)
        {
            elements[i] -= otherVector.elements[i];
        }
        return *this;
    }

    Vector &operator*=(const Vector &otherVector)
    {
        if (size != otherVector.size)
        {
            throw std::runtime_error("Vector sizes are not compatible for compound multiplication");
        }

        // This implementation assumes element-wise multiplication.
        for (size_t i = 0; i < size; ++i)
        {
            elements[i] *= otherVector.elements[i];
        }
        return *this;
    }

    bool operator==(const Vector &otherVector) const
    {
        if (size != otherVector.size)
        {
            return false;
        }

        for (size_t i = 0; i < size; ++i)
        {
            if (elements[i] != otherVector.elements[i])
            {
                return false;
            }
        }
        return true;
    }
};

class SquareMatrix : public Matrix
{
public:
    SquareMatrix(size_t size) : Matrix(size, size) {}

    // Get the diagonal elements of the square matrix
    std::vector<Complex> getDiagonal() const
    {
        std::vector<Complex> diagonalElements;
        for (size_t i = 0; i < getRows(); ++i)
        {
            diagonalElements.push_back(at(i, i));
        }
        return diagonalElements;
    }

    // Check if the square matrix is symmetric
    bool isSymmetric() const
    {
        if (getRows() != getCols())
        {
            return false; // Square matrices only can be symmetric
        }

        for (size_t i = 0; i < getRows(); ++i)
        {
            for (size_t j = i + 1; j < getCols(); ++j)
            {
                if (at(i, j) != at(j, i))
                {
                    return false;
                }
            }
        }
        return true;
    }

    // Method to calculate the trace of the square matrix
    Complex trace() const
    {
        if (getRows() != getCols())
        {
            throw std::runtime_error("Trace can only be calculated for square matrices");
        }

        Complex traceSum(0, 0);
        for (size_t i = 0; i < getRows(); ++i)
        {
            traceSum += at(i, i);
        }
        return traceSum;
    }
};

class IdentityMatrix : public SquareMatrix
{
public:
    IdentityMatrix(size_t size) : SquareMatrix(size)
    {
        // Implement logic to set up an identity matrix during construction
        for (size_t i = 0; i < size; ++i)
        {
            for (size_t j = 0; j < size; ++j)
            {
                if (i == j)
                {
                    at(i, j) = Complex(1, 0); // Diagonal elements are 1
                }
                else
                {
                    at(i, j) = Complex(0, 0); // Non-diagonal elements are 0
                }
            }
        }
    }

    // Method to check if the matrix is an identity matrix
    bool isIdentityMatrix() const
    {
        for (size_t i = 0; i < getRows(); ++i)
        {
            for (size_t j = 0; j < getCols(); ++j)
            {
                if (i == j && at(i, j) != Complex(1, 0))
                {
                    return false; // Diagonal elements must be 1
                }
                else if (i != j && at(i, j) != Complex(0, 0))
                {
                    return false; // Non-diagonal elements must be 0
                }
            }
        }
        return true; // Matrix is an identity matrix
    }
};

class TriangleMatrix : public Matrix
{
public:
    TriangleMatrix(size_t row, size_t col) : Matrix(row, col)
    {
        // Call the base class constructor to set up the matrix
        // Matrix(row, col) already creates the necessary matrix structure

        // Now, initialize the triangle matrix during construction
        // For this example, let's assume we are working with an upper triangle matrix
        // Modify this logic based on whether it's an upper or lower triangle matrix
        for (size_t i = 0; i < row; ++i)
        {
            for (size_t j = 0; j < col; ++j)
            {
                if (i <= j)
                {
                    // Initialize only the upper triangle elements, rest would remain default (zeros)
                    at(i, j) = Complex(1.0, 2.0); // Replace 1.0 and 2.0 with your desired real and imaginary values
                }
            }
        }
    }

    // Method to check if the matrix is an upper triangle matrix
    bool isUpperTriangle() const
    {
        if (getRows() != getCols())
        {
            return false; // Triangle matrices are square matrices
        }

        for (size_t i = 0; i < getRows(); ++i)
        {
            for (size_t j = 0; j < i; ++j)
            {
                if (at(i, j) != Complex(0, 0))
                {
                    return false;
                }
            }
        }
        return true;
    }

    // Method to check if the matrix is a lower triangle matrix
    bool isLowerTriangle() const
    {
        if (getRows() != getCols())
        {
            return false; // Triangle matrices are square matrices
        }

        for (size_t i = 0; i < getRows(); ++i)
        {
            for (size_t j = i + 1; j < getCols(); ++j)
            {
                if (at(i, j) != Complex(0, 0))
                {
                    return false;
                }
            }
        }
        return true;
    }

    // Method to get the diagonal elements of the triangle matrix
    std::vector<Complex> getDiagonal() const
    {
        std::vector<Complex> diagonalElements;
        for (size_t i = 0; i < getRows(); ++i)
        {
            diagonalElements.push_back(at(i, i));
        }
        return diagonalElements;
    }
};

class LinearAlgebraObject
{
public:
    virtual void print() const = 0;                              // Method to print the object
    virtual size_t getRows() const = 0;                          // Method to get the number of rows
    virtual size_t getCols() const = 0;                          // Method to get the number of columns
    virtual Complex &at(size_t row, size_t col) = 0;             // Method to access an element
    virtual const Complex &at(size_t row, size_t col) const = 0; // Const version to access an element
    virtual LinearAlgebraObject &operator=(const LinearAlgebraObject &other) = 0;

    // Pure virtual copy constructor
    virtual LinearAlgebraObject *clone() const = 0;
};

int main()
{
    // Test Complex numbers
    Complex c1(2, 3);
    Complex c2(4, 5);

    Complex c3 = c1 + c2;
    Complex c4 = c1 - c2;
    Complex c5 = c1 * c2;
    Complex c6 = c1 / c2;

    // Displaying results
    std::cout << "c3 = (c1 + c2): " << c3.getReal() << " + " << c3.getImaginary() << "i" << std::endl;
    std::cout << "c4 = (c1 - c2): " << c4.getReal() << " + " << c4.getImaginary() << "i" << std::endl;
    std::cout << "c5 = (c1 * c2): " << c5.getReal() << " + " << c5.getImaginary() << "i" << std::endl;
    std::cout << "c6 = (c1 / c2): " << c6.getReal() << " + " << c6.getImaginary() << "i" << std::endl;

    // Test Vectors
    Vector v1(3);
    v1.at(0) = Complex(1, 2);
    v1.at(1) = Complex(3, 4);
    v1.at(2) = Complex(5, 6);

    Vector v2(3);
    v2.at(0) = Complex(2, 3);
    v2.at(1) = Complex(4, 5);
    v2.at(2) = Complex(6, 7);

    Vector v3 = v1 + v2;
    Vector v4 = v1 - v2;
    Complex v5 = v1 * v2; // Assuming you meant dot product in the comment

    // Displaying results
    std::cout << "\nVector v3 = (v1 + v2): ";
    for (size_t i = 0; i < v3.getSize(); ++i)
    {
        std::cout << v3.at(i).getReal() << " + " << v3.at(i).getImaginary() << "i\t";
    }
    std::cout << std::endl;

    std::cout << "Vector v4 = (v1 - v2): ";
    for (size_t i = 0; i < v4.getSize(); ++i)
    {
        std::cout << v4.at(i).getReal() << " + " << v4.at(i).getImaginary() << "i\t";
    }
    std::cout << std::endl;

    std::cout << "v5 = (v1 * v2 - dot product): " << v5.getReal() << " + " << v5.getImaginary() << "i" << std::endl;

    // Test Matrices, SquareMatrix, IdentityMatrix, TriangleMatrix, etc.
    // Create instances and perform operations

    // Test Matrix
    Matrix mat1(2, 2);
    mat1.at(0, 0) = Complex(1, 2);
    mat1.at(0, 1) = Complex(3, 4);
    mat1.at(1, 0) = Complex(5, 6);
    mat1.at(1, 1) = Complex(7, 8);

    std::cout << "\nMatrix 1:" << std::endl;
    mat1.print();

    Matrix mat2(2, 2);
    mat2.at(0, 0) = Complex(2, 3);
    mat2.at(0, 1) = Complex(4, 5);
    mat2.at(1, 0) = Complex(6, 7);
    mat2.at(1, 1) = Complex(8, 9);

    std::cout << "\nMatrix 2:" << std::endl;
    mat2.print();

    // Test matrix addition
    Matrix mat3 = mat1 + mat2;
    std::cout << "\nMatrix Addition (Matrix 1 + Matrix 2):" << std::endl;
    mat3.print();

    // Test matrix subtraction
    Matrix mat4 = mat1 - mat2;
    std::cout << "\nMatrix subtraction (Matrix 1 - Matrix 2):" << std::endl;
    mat4.print();

    // Test matrix multiplication
    Matrix mat6 = mat1 * mat2;
    std::cout << "\nMatrix Multiplication (Matrix 1 * Matrix 2):" << std::endl;
    mat6.print();

    // Test operator= for Matrix
    Matrix mat5 = mat4; // Using the copy constructor
    std::cout << "\nTesting operator= for Matrix:" << std::endl;
    mat5.print();

    // Test operator+= for Matrix
    std::cout << "\nTesting operator+= for Matrix:" << std::endl;
    mat5 += mat2; // Adding mat2 to mat5
    mat5.print();
    // Test operator-= for Matrix
    std::cout << "\nTesting operator-= for Matrix:" << std::endl;
    mat5 -= mat2; // Subtracting mat2 from mat5 (restoring original value)
    mat5.print();

    // Test operator*= for Matrix
    std::cout << "\nTesting operator*= for Matrix:" << std::endl;
    mat5 *= mat2; // Multiplying mat5 by mat2
    mat5.print();

    // Test operator== for Matrix
    std::cout << "\nTesting operator== for Matrix:" << std::endl;
    if (mat1 == mat5)
    {
        std::cout << "Matrix 1 is equal to modified Matrix (mat5)" << std::endl;
    }
    else
    {
        std::cout << "Matrices are not equal" << std::endl;
    }

    // Test operator+ for Vector
    std::cout << "\nTesting operator+ for Vector:" << std::endl;
    Vector v7 = v1 + v2; // Adding v1 and v2 using operator+
    for (size_t i = 0; i < v7.getSize(); ++i)
    {
        std::cout << v7.at(i).getReal() << " + " << v7.at(i).getImaginary() << "i\t";
    }
    std::cout << std::endl;

    // Test operator- for Vector
    std::cout << "\nTesting operator- for Vector:" << std::endl;
    Vector v8 = v1 - v2; // Subtracting v2 from v1 using operator-
    for (size_t i = 0; i < v8.getSize(); ++i)
    {
        std::cout << v8.at(i).getReal() << " + " << v8.at(i).getImaginary() << "i\t";
    }
    std::cout << std::endl;

    // Test operator* for Vector (assuming dot product)
    std::cout << "\nTesting operator* for Vector (dot product):" << std::endl;
    Complex dotProductVector = v1 * v2; // Calculating dot product using operator*
    std::cout << "Dot Product (v1 * v2): " << dotProductVector.getReal() << " + " << dotProductVector.getImaginary() << "i" << std::endl;

    // Test operator= for Vector
    Vector v6(3);
    v6 = v1; // Using the assignment operator
    std::cout << "\nTesting operator= for Vector:" << std::endl;
    for (size_t i = 0; i < v6.getSize(); ++i)
    {
        std::cout << v6.at(i).getReal() << " + " << v6.at(i).getImaginary() << "i\t";
    }
    std::cout << std::endl;

    // Test operator+= and operator-= for Vector
    std::cout << "\nTesting operator+= for Vector:" << std::endl;
    v6 += v2; // Adding v2 to v6
    for (size_t i = 0; i < v6.getSize(); ++i)
    {
        std::cout << v6.at(i).getReal() << " + " << v6.at(i).getImaginary() << "i\t";
    }
    std::cout << std::endl;

    std::cout << "\nTesting operator-= for Vector:" << std::endl;
    v6 -= v2; // Subtracting v2 from v6 (restoring original value)
    for (size_t i = 0; i < v6.getSize(); ++i)
    {
        std::cout << v6.at(i).getReal() << " + " << v6.at(i).getImaginary() << "i\t";
    }
    std::cout << std::endl;

    // Test operator*= for Vector (considering dot product)
    std::cout << "\nTesting operator*= for Vector (dot product):" << std::endl;
    Complex dotProduct = v1 * v2;
    std::cout << "Dot Product (v1 * v2): " << dotProduct.getReal() << " + " << dotProduct.getImaginary() << "i" << std::endl;

    // Test operator== for Vector
    std::cout << "\nTesting operator== for Vector:" << std::endl;
    if (v1 == v6)
    {
        std::cout << "Vectors v1 and v6 are equal" << std::endl;
    }
    else
    {
        std::cout << "Vectors are not equal" << std::endl;
    }

    // Test SquareMatrix
    SquareMatrix sqMat(3);
    sqMat.at(0, 0) = Complex(1, 0);
    sqMat.at(0, 1) = Complex(2, 0);
    sqMat.at(0, 2) = Complex(3, 0);
    sqMat.at(1, 0) = Complex(4, 0);
    sqMat.at(1, 1) = Complex(5, 0);
    sqMat.at(1, 2) = Complex(6, 0);
    sqMat.at(2, 0) = Complex(7, 0);
    sqMat.at(2, 1) = Complex(8, 0);
    sqMat.at(2, 2) = Complex(9, 0);

    std::cout << "\nSquare Matrix:" << std::endl;
    sqMat.print();

    // Calculate the determinant
    Complex det = sqMat.determinant();
    std::cout << "Determinant: " << det.getReal() << "+" << det.getImaginary() << "i" << std::endl;

    // Check if the matrix is symmetric
    bool isSymmetric = sqMat.isSymmetric();
    std::cout << "Is symmetric: " << (isSymmetric ? "Yes" : "No") << std::endl;

    // Get the diagonal elements
    std::vector<Complex> diagonalElements = sqMat.getDiagonal();
    std::cout << "Diagonal Elements:" << std::endl;
    for (const auto &element : diagonalElements)
    {
        std::cout << element.getReal() << "+" << element.getImaginary() << "i" << std::endl;

        // Test IdentityMatrix
        IdentityMatrix idMat(4);
        bool isIdentity = idMat.isIdentityMatrix();
        std::cout << "Is identity matrix: " << (isIdentity ? "Yes" : "No") << std::endl;
        idMat.print();

        // Test TriangleMatrix
        // Create a TriangleMatrix
        std::cout << "\nTriangle Matrix :" << std::endl;
        TriangleMatrix upperTriangleMat(4, 4);

        // Fill upper triangle with values
        upperTriangleMat.at(0, 0) = Complex(1, 0);
        upperTriangleMat.at(0, 1) = Complex(2, 0);
        upperTriangleMat.at(0, 2) = Complex(3, 0);
        upperTriangleMat.at(0, 3) = Complex(4, 0);
        upperTriangleMat.at(1, 1) = Complex(5, 0);
        upperTriangleMat.at(1, 2) = Complex(6, 0);
        upperTriangleMat.at(1, 3) = Complex(7, 0);
        upperTriangleMat.at(2, 2) = Complex(8, 0);
        upperTriangleMat.at(2, 3) = Complex(9, 0);
        upperTriangleMat.at(3, 3) = Complex(10, 0);

        upperTriangleMat.print();

        // Check if the matrix is an upper triangle
        bool isUpperTriangle = upperTriangleMat.isUpperTriangle();
        std::cout << "Is upper triangle matrix: " << (isUpperTriangle ? "Yes" : "No") << std::endl;

        // Check if the matrix is a lower triangle
        bool isLowerTriangle = upperTriangleMat.isLowerTriangle();
        std::cout << "Is lower triangle matrix: " << (isLowerTriangle ? "Yes" : "No") << std::endl;

        // Get the diagonal elements
        std::vector<Complex> diagonalElems = upperTriangleMat.getDiagonal();
        std::cout << "Diagonal Elements:" << std::endl;
        for (const auto &elem : diagonalElems)
        {
            std::cout << elem.getReal() << "+" << elem.getImaginary() << "i" << std::endl;
        }

        return 0;
    }
};
